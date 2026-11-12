"""`triager batch`: run a matrix of triages, for a corpus a recorder can study.

A plan (YAML) lists groups: which cases (`real` or `synthetic`, optionally a sample), which
model, how many repeats, which tool failures to inject, and a pause between runs for
rate-limited providers. Each run is written to a manifest (JSON lines) as it finishes, so a
batch that stops (budget, Ctrl+C, a crash) carries on where it left off. Runs that ended in
an error are tried again on resume.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from flight_recorder.llm import ModelSpecError, parse_spec

from issue_triager.dataset import Dataset, TriageCase
from issue_triager.tools import FAILURE_MODES, FailureMode

CaseSet = Literal["real", "synthetic"]
MAX_TRIES = 6  # attempts per run while rate-limited
FIRST_BACKOFF = 15.0  # seconds; doubles each time
RATE_LIMIT_SIGNS = ("429", "rate limit", "ratelimit", "resource_exhausted", "resourceexhausted",
                    "quota")  # fmt: skip


BILLING_SIGNS = ("credit balance", "insufficient_quota", "billing", "payment required")


class PlanError(ValueError):
    """The plan file is wrong; the message says how."""


class RunError(Exception):
    """A run that raised, with the trace it was recorded under (if any)."""

    def __init__(self, cause: BaseException, trace_id: str | None) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.trace_id = trace_id


@dataclass(frozen=True)
class Group:
    name: str
    cases: CaseSet
    model: str
    comment_model: str | None = None
    repeats: int = 1
    sample: int | None = None
    inject: tuple[FailureMode, ...] = ()
    rate: float = 0.3
    pause_seconds: float = 0.0


@dataclass(frozen=True)
class Plan:
    groups: tuple[Group, ...]
    budget_usd: float | None = None


@dataclass(frozen=True)
class Job:
    group: Group
    case: TriageCase
    repeat: int
    seed: int

    @property
    def key(self) -> str:
        return f"{self.group.name}:{self.case.id}:{self.repeat}"


@dataclass(frozen=True)
class Outcome:
    status: str  # applied, rejected, failed (validation) or error
    grade: str | None = None  # correct, wrong, failed; None after an error
    cost_usd: float | None = None
    trace_id: str | None = None
    error: str | None = None
    injected: tuple[str, ...] = ()


@dataclass
class Summary:
    done: int = 0
    skipped: int = 0
    errors: int = 0
    cost_usd: float = 0.0
    grades: dict[str, int] = field(default_factory=lambda: {})
    stopped: str | None = None  # "budget" or "billing"; None if every job ran


def load_plan(path: Path) -> Plan:
    """Read and check a plan file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("groups"), list):
        raise PlanError(f"{path}: needs a `groups:` list")
    plan = cast("dict[str, Any]", raw)
    groups = tuple(_group(cast("dict[str, Any]", g)) for g in plan["groups"])
    if len({g.name for g in groups}) != len(groups):
        raise PlanError("group names must be unique")
    budget = plan.get("budget_usd")
    return Plan(groups, float(budget) if budget is not None else None)


def _group(raw: dict[str, Any]) -> Group:
    name = str(raw.get("name") or "")
    if not name:
        raise PlanError("every group needs a name")
    cases = raw.get("cases")
    if cases not in ("real", "synthetic"):
        raise PlanError(f"group {name}: cases must be real or synthetic, not {cases!r}")
    model = raw.get("model")
    if not model:
        raise PlanError(f"group {name}: needs a model")
    for spec in (model, raw.get("comment_model")):
        if spec is not None:
            try:
                parse_spec(str(spec))
            except ModelSpecError as error:
                raise PlanError(f"group {name}: model {spec!r}: {error}") from error
    inject = tuple(str(mode) for mode in raw.get("inject") or ())
    unknown = [mode for mode in inject if mode not in FAILURE_MODES]
    if unknown:
        raise PlanError(f"group {name}: unknown inject {unknown}; choose from {FAILURE_MODES}")
    return Group(
        name=name,
        cases=cast("CaseSet", cases),
        model=str(model),
        comment_model=raw.get("comment_model"),
        repeats=int(raw.get("repeats", 1)),
        sample=int(raw["sample"]) if raw.get("sample") is not None else None,
        inject=cast("tuple[FailureMode, ...]", inject),
        rate=float(raw.get("rate", 0.3)),
        pause_seconds=float(raw.get("pause_seconds", 0)),
    )


def _stable(text: str) -> int:
    """A seed that's the same on every machine and run (unlike hash())."""
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def plan_jobs(plan: Plan, data: Dataset) -> list[Job]:
    """Every run the plan asks for, in order: group by group, case by case, repeat by repeat.
    A sample is drawn with a seed from the group's name, so it's the same every time."""
    jobs: list[Job] = []
    for group in plan.groups:
        cases = list(data.real if group.cases == "real" else data.synthetic)
        if group.sample is not None and group.sample < len(cases):
            cases = random.Random(_stable(group.name)).sample(cases, group.sample)
        for case in cases:
            for repeat in range(1, group.repeats + 1):
                seed = _stable(f"{group.name}:{case.id}:{repeat}")
                jobs.append(Job(group, case, repeat, seed))
    return jobs


def _signs(error: BaseException, signs: Sequence[str]) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return any(sign in text for sign in signs)


def is_rate_limited(error: BaseException) -> bool:
    return _signs(error, RATE_LIMIT_SIGNS)


def is_billing(error: BaseException) -> bool:
    """Out of credit or quota: every later run would fail the same way."""
    return _signs(error, BILLING_SIGNS)


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """The latest line for each key: a key tried again after an error appears twice."""
    if not path.exists():
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            entries[entry["key"]] = entry
    return entries


def run_batch(
    jobs: Sequence[Job],
    triage: Callable[[Job], Outcome],
    *,
    manifest: Path,
    budget_usd: float | None,
    sleep: Callable[[float], None] = time.sleep,
    on_done: Callable[[Job, Outcome], None] | None = None,
) -> Summary:
    """Run the jobs not yet done in the manifest, writing each as it finishes."""
    done = {k: e for k, e in _read_manifest(manifest).items() if e["status"] != "error"}
    summary = Summary(cost_usd=sum(e.get("cost_usd") or 0.0 for e in done.values()))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pending = [job for job in jobs if job.key not in done]
    summary.skipped = len(jobs) - len(pending)
    for index, job in enumerate(pending):
        if budget_usd is not None and summary.cost_usd >= budget_usd:
            summary.stopped = "budget"
            break
        outcome, billing = _attempt(job, triage, sleep)
        _append(manifest, job, outcome)
        summary.cost_usd += outcome.cost_usd or 0.0
        if outcome.status == "error":
            summary.errors += 1
        else:
            summary.done += 1
            if outcome.grade:
                summary.grades[outcome.grade] = summary.grades.get(outcome.grade, 0) + 1
        if on_done:
            on_done(job, outcome)
        if billing:
            summary.stopped = "billing"
            break
        more = index + 1 < len(pending)
        if more and job.group.pause_seconds and pending[index + 1].group == job.group:
            sleep(job.group.pause_seconds)
    return summary


def _attempt(
    job: Job, triage: Callable[[Job], Outcome], sleep: Callable[[float], None]
) -> tuple[Outcome, bool]:
    """One run, backing off and trying again while the provider is rate-limiting. Returns
    the outcome and whether it was a billing error (out of credit: stop the batch)."""
    wait = FIRST_BACKOFF
    for attempt in range(1, MAX_TRIES + 1):
        try:
            return triage(job), False
        except Exception as raised:  # provider SDKs raise their own types
            error = raised.cause if isinstance(raised, RunError) else raised
            trace_id = raised.trace_id if isinstance(raised, RunError) else None
            if is_rate_limited(error) and not is_billing(error) and attempt < MAX_TRIES:
                sleep(wait)
                wait *= 2
                continue
            first = str(error).splitlines()[0] if str(error) else ""
            message = f"{type(error).__name__}: {first}"[:500]
            return Outcome(status="error", error=message, trace_id=trace_id), is_billing(error)
    raise AssertionError("unreachable")


def _append(manifest: Path, job: Job, outcome: Outcome) -> None:
    entry = {
        "key": job.key,
        "group": job.group.name,
        "case": job.case.id,
        "source": job.case.source,
        "repeat": job.repeat,
        "model": job.group.model,
        "inject": list(job.group.inject),
        "seed": job.seed,
        **asdict(outcome),
        "injected": list(outcome.injected),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with manifest.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")
