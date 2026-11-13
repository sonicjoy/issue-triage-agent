"""Datasets: one GitHub repo's issues, its label/milestone taxonomy, and optional extras.

A dataset is a folder:

    repo.json       exported: description, labels, milestones
    issues.json     exported: every issue, with its real triage as ground truth
    synthetic.yaml  optional, hand-written: extra cases (source: synthetic)
    rules.yaml      optional, hand-written: repo-specific triage rules

Each case pairs what the agent may see (`TriageInput`) with the answer it is
graded against (`GroundTruth`). Keep the two apart: the agent must never be
given the target issue's own labels or milestone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DATASETS_DIR = Path(__file__).parent / "datasets"
DEFAULT_DATASET = DATASETS_DIR / "sonicjoy__agent-flight-recorder"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Label(_Frozen):
    name: str
    description: str = ""


class Milestone(_Frozen):
    title: str
    description: str = ""
    due_on: str | None = None
    state: Literal["open", "closed"] = "open"


class RepoInfo(_Frozen):
    """What a triager may know about the repo: its purpose and its taxonomy."""

    repo: str
    description: str = ""
    labels: list[Label] = Field(default_factory=list[Label])
    milestones: list[Milestone] = Field(default_factory=list[Milestone])

    @property
    def label_names(self) -> frozenset[str]:
        return frozenset(label.name for label in self.labels)

    @property
    def milestone_titles(self) -> frozenset[str]:
        return frozenset(m.title for m in self.milestones)


class Rules(_Frozen):
    """Repo-specific rules on top of the taxonomy (rules.yaml)."""

    require_label: bool = False
    labels_needing_milestone: frozenset[str] = frozenset()
    """For example `stretch` ("cut first from its milestone") is meaningless without one."""


class TriageInput(_Frozen):
    """What the agent is given about the issue it triages."""

    id: str
    title: str
    body: str


TRUTH_FIELDS = ("labels", "milestone", "duplicate_of", "regression_of")


class GroundTruth(_Frozen):
    """The triage decision the agent is graded against.

    `duplicate_of` and `regression_of` both point at an earlier issue, but mean
    different things and call for different actions:

    - duplicate: asks for the same work again. Close it, link the original.
    - regression: reports that work an issue already delivered (closed as
      completed) has broken again. Keep it open as a bug; link the original.
    """

    labels: frozenset[str]
    milestone: str | None
    duplicate_of: str | None = None
    regression_of: str | None = None
    accept: dict[str, list[Any]] = Field(default_factory=dict[str, list[Any]])
    """Other answers that also count as right, per field, for a debatable truth (AFR #14):
    `{"duplicate_of": [None]}`. Grading compares them like the truth."""

    @model_validator(mode="after")
    def _one_relation(self) -> Self:
        if self.duplicate_of and self.regression_of:
            raise ValueError("an issue is a duplicate or a regression, not both")
        unknown = sorted(set(self.accept) - set(TRUTH_FIELDS))
        if unknown:
            raise ValueError(f"accept: {unknown} aren't truth fields ({', '.join(TRUTH_FIELDS)})")
        return self

    def __hash__(self) -> int:  # accept is a dict; the rest identifies the truth
        return hash((self.labels, self.milestone, self.duplicate_of, self.regression_of))


class TriageCase(_Frozen):
    id: str
    """Real issues use their number ("7"); synthetic cases use "s01", "s02", ..."""
    title: str
    body: str
    truth: GroundTruth
    source: Literal["real", "synthetic"]
    state: Literal["open", "closed"] = "open"
    kind: Literal["issue", "paraphrase", "tricky"] = "issue"
    based_on: str | None = None
    """For paraphrases: the real issue this rewords."""
    note: str = ""
    """For tricky cases: what makes the case hard."""
    truth_override: str = ""
    """Why the exported truth was corrected (truth_overrides.yaml), or empty."""

    def agent_input(self) -> TriageInput:
        return TriageInput(id=self.id, title=self.title, body=self.body)


class IssueExport(_Frozen):
    repo: str
    issues: list[TriageCase] = Field(default_factory=list[TriageCase])


@dataclass(frozen=True)
class Dataset:
    path: Path
    repo: RepoInfo
    real: list[TriageCase]
    synthetic: list[TriageCase] = field(default_factory=list[TriageCase])
    rules: Rules = field(default_factory=Rules)

    @property
    def cases(self) -> list[TriageCase]:
        """All cases, real first. Report results per `source`."""
        return [*self.real, *self.synthetic]

    @cached_property
    def by_id(self) -> Mapping[str, TriageCase]:
        return {c.id: c for c in self.cases}

    @cached_property
    def corpus(self) -> Mapping[str, TriageCase]:
        """What the agent can search: real issues only."""
        return {c.id: c for c in self.real}

    def problems(self, truth: GroundTruth, *, target_id: str) -> list[str]:
        """Rule violations, worded as feedback an agent can act on. Empty means valid."""
        found: list[str] = []
        unknown = sorted(truth.labels - self.repo.label_names)
        if unknown:
            found.append(f"Unknown labels {unknown}; use only the repo's labels.")
        if self.rules.require_label and not truth.labels:
            found.append("Pick at least one label.")
        if truth.milestone is not None and truth.milestone not in self.repo.milestone_titles:
            found.append(f"Unknown milestone {truth.milestone!r}; use a listed one or null.")
        needing = sorted(truth.labels & self.rules.labels_needing_milestone)
        if needing and truth.milestone is None:
            found.append(f"Labels {needing} need a milestone.")
        refs = (("duplicate_of", truth.duplicate_of), ("regression_of", truth.regression_of))
        for name, ref in refs:
            if ref is None:
                continue
            ref = ref.lstrip("#")
            original = self.corpus.get(ref)
            if ref == target_id:
                found.append(f"{name} can't point at the issue being triaged.")
            elif original is None:
                found.append(f"{name}: issue #{ref} doesn't exist.")
            elif name == "regression_of" and (
                original.state != "closed" or original.truth.duplicate_of
            ):
                found.append(
                    f"regression_of: #{ref} isn't done, so it can't have regressed. "
                    "A bug in unfinished work is neither a duplicate nor a regression."
                )
        return found


def load_dataset(path: Path = DEFAULT_DATASET) -> Dataset:
    repo = RepoInfo.model_validate_json((path / "repo.json").read_text(encoding="utf-8"))
    issues = IssueExport.model_validate_json((path / "issues.json").read_text(encoding="utf-8"))
    synthetic: list[TriageCase] = []
    if (path / "synthetic.yaml").exists():
        raw = yaml.safe_load((path / "synthetic.yaml").read_text(encoding="utf-8"))
        synthetic = [
            TriageCase.model_validate({**case, "source": "synthetic"}) for case in raw["cases"]
        ]
    rules = Rules()
    if (path / "rules.yaml").exists():
        rules = Rules.model_validate(yaml.safe_load((path / "rules.yaml").read_text("utf-8")))
    real = issues.issues
    if (path / "truth_overrides.yaml").exists():
        raw = yaml.safe_load((path / "truth_overrides.yaml").read_text(encoding="utf-8"))
        real, synthetic = _overridden(real, synthetic, (raw or {}).get("overrides") or {})
    return Dataset(path=path, repo=repo, real=real, synthetic=synthetic, rules=rules)


def _overridden(
    real: list[TriageCase], synthetic: list[TriageCase], overrides: dict[Any, Any]
) -> tuple[list[TriageCase], list[TriageCase]]:
    """Real cases with corrected truths (truth_overrides.yaml, AFR #14), and paraphrases of
    them following suit: a paraphrase's truth is its source's triage, plus its own link."""
    fixed = {case.id: case for case in real}
    for key, fix in overrides.items():
        issue, changes = str(key), dict(fix or {})
        if issue not in fixed:
            raise ValueError(f"truth_overrides: #{issue} isn't a real issue in the export")
        reason = str(changes.pop("reason", "") or "").strip()
        if not reason:
            raise ValueError(f"truth_overrides: #{issue} needs a reason")
        unknown = sorted(set(changes) - set(TRUTH_FIELDS))
        if unknown:
            raise ValueError(f"truth_overrides: #{issue}: {unknown} aren't truth fields")
        case = fixed[issue]
        truth = GroundTruth.model_validate(case.truth.model_dump() | changes)
        fixed[issue] = case.model_copy(update={"truth": truth, "truth_override": reason})
    following: list[TriageCase] = []
    for case in synthetic:
        source = fixed.get(str(case.based_on)) if case.kind == "paraphrase" else None
        if source is not None and source.truth_override:
            own = {"duplicate_of": case.truth.duplicate_of, "accept": case.truth.accept}
            truth = GroundTruth.model_validate(source.truth.model_dump() | own)
            case = case.model_copy(update={"truth": truth, "truth_override": source.truth_override})
        following.append(case)
    return [fixed[case.id] for case in real], following


def dataset_dir_name(repo: str) -> str:
    """'owner/name' -> 'owner__name'."""
    return repo.replace("/", "__")


def _dump(data: object) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def dump_repo(repo: RepoInfo) -> str:
    return _dump(repo.model_dump(mode="json", exclude_defaults=True))


def dump_export(export: IssueExport) -> str:
    """Stable JSON, so re-exporting an unchanged repo leaves no diff."""
    data = export.model_dump(mode="json", exclude_defaults=True)
    for issue in data["issues"]:
        issue["truth"]["labels"] = sorted(issue["truth"]["labels"])
    return _dump(data)
