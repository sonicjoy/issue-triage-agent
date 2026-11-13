"""`triager batch` and grades on the trace (recorder #13), written before the code.

Decisions (agent-flight-recorder/.review/13-corpus/plan.md):
- a matrix of cases x models x repeats x injection, recorded, resumable;
- each run's grade against the ground truth goes on its root span, never into the prompt;
- a spend cap stops the batch cleanly;
- Gemini runs are paced and back off on rate limits;
- a local manifest records every run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from test_recording import Collector, collector  # noqa: F401  (fixture)
from test_triager import DATA, call, decide
from test_triager import models as langgraph_models
from typer.testing import CliRunner

from issue_triager import cli

REAL_19 = DATA.by_id["19"]


def plan_text(**overrides: Any) -> str:
    groups = overrides.pop("groups", None) or [
        {"name": "real-main", "cases": "real", "model": "anthropic:claude-sonnet-5-5",
         "repeats": 2},
        {"name": "synthetic-cheap", "cases": "synthetic", "model": "anthropic:claude-haiku-5-5"},
        {"name": "injected", "cases": "real", "sample": 5, "model": "anthropic:claude-sonnet-5-5",
         "inject": ["timeout", "bad-data"], "rate": 0.5},
        {"name": "gemini", "cases": "real", "model": "google_genai:gemini-3.5-flash-lite",
         "pause_seconds": 4},
    ]  # fmt: skip
    return json.dumps({"budget_usd": 10, "groups": groups} | overrides)  # JSON is valid YAML


def write_plan(tmp_path: Path, **overrides: Any) -> Path:
    path = tmp_path / "plan.yaml"
    path.write_text(plan_text(**overrides), encoding="utf-8")
    return path


# ---- grading ---------------------------------------------------------------------


def test_grade_fields_compare_each_field_with_the_ground_truth() -> None:
    from issue_triager.grading import grade_fields

    truth = REAL_19.truth
    right = {"labels": list(truth.labels), "milestone": truth.milestone,
             "duplicate_of": truth.duplicate_of, "regression_of": truth.regression_of}  # fmt: skip
    assert grade_fields(right, truth) == dict.fromkeys(
        ("labels", "milestone", "duplicate_of", "regression_of"), True
    )
    wrong = right | {"labels": ["ui"], "duplicate_of": "#2"}
    assert grade_fields(wrong, truth) == {"labels": False, "milestone": True,
                                          "duplicate_of": truth.duplicate_of == "2",
                                          "regression_of": True}  # fmt: skip


def test_grade_attributes_say_correct_wrong_or_failed() -> None:
    from issue_triager.grading import grade_attributes

    truth = REAL_19.truth
    right = {"labels": list(truth.labels), "milestone": truth.milestone,
             "duplicate_of": truth.duplicate_of, "regression_of": truth.regression_of}  # fmt: skip

    correct = grade_attributes({"status": "applied", "decision": right}, REAL_19)
    assert correct["triager.grade"] == "correct"
    assert correct["triager.grade.labels"] == "ok"
    assert correct["triager.case_source"] == "real"

    wrong = grade_attributes({"status": "applied", "decision": right | {"labels": []}}, REAL_19)
    assert wrong["triager.grade"] == "wrong" and wrong["triager.grade.labels"] == "wrong"

    failed = grade_attributes({"status": "failed", "decision": None}, REAL_19)
    assert failed["triager.grade"] == "failed"
    assert "triager.grade.labels" not in failed


def test_grade_attributes_show_what_was_decided_and_what_was_expected() -> None:
    """#14: a labeller reads the decision next to the truth, so both go on the root span."""
    from issue_triager.grading import grade_attributes

    truth = REAL_19.truth
    decision = {"labels": ["infra", "evals"], "milestone": None, "duplicate_of": "#3",
                "regression_of": None}  # fmt: skip

    marks = grade_attributes({"status": "applied", "decision": decision}, REAL_19)

    assert marks["triager.decided.labels"] == ["evals", "infra"]  # sorted, a string list
    assert marks["triager.decided.milestone"] == "(none)"
    assert marks["triager.decided.duplicate_of"] == "3"  # compared without the #
    assert marks["triager.decided.regression_of"] == "(none)"
    assert marks["triager.expected.labels"] == sorted(truth.labels)
    assert marks["triager.expected.milestone"] == (truth.milestone or "(none)")
    assert marks["triager.expected.duplicate_of"] == (truth.duplicate_of or "(none)")
    assert marks["triager.expected.regression_of"] == (truth.regression_of or "(none)")


def test_a_failed_run_shows_only_what_was_expected() -> None:
    from issue_triager.grading import grade_attributes

    failed = grade_attributes({"status": "failed", "decision": None}, REAL_19)

    assert failed["triager.expected.labels"] == sorted(REAL_19.truth.labels)
    assert not any(key.startswith("triager.decided.") for key in failed)


def test_the_root_span_carries_the_grade(
    clean_env: pytest.MonkeyPatch,
    tmp_path: Path,
    collector: Collector,  # noqa: F811
) -> None:
    fake = langgraph_models(iter([call("search_issues", {"query": "evals"}), decide()]))
    clean_env.setattr(cli, "resolve_models", lambda *a, **k: fake)
    result = CliRunner().invoke(
        cli.app,
        ["run", "--issue", "19", "--yes", "--runs-dir", str(tmp_path), "--record", collector.url],
    )

    assert result.exit_code == 0, result.output
    spans = [a for _, a in collector.spans()]
    [root] = [a for a in spans if a["name"].startswith("triage ")]
    assert root["triager.grade"] in ("correct", "wrong")
    assert root["triager.grade.labels"] in ("ok", "wrong")
    assert root["triager.triage_model"] == "openai:gpt-5.4-nano"
    assert root["triager.expected.milestone"] == REAL_19.truth.milestone
    assert "triager.decided.labels" in root
    child_text = json.dumps([a for a in spans if a is not root])
    assert "triager.grade" not in child_text  # the agent's own spans never see the truth
    assert "triager.expected" not in child_text


# ---- the plan --------------------------------------------------------------------


def test_the_plan_expands_to_jobs(tmp_path: Path) -> None:
    from issue_triager.batch import load_plan, plan_jobs

    jobs = plan_jobs(load_plan(write_plan(tmp_path)), DATA)
    by_group: dict[str, list[Any]] = {}
    for job in jobs:
        by_group.setdefault(job.group.name, []).append(job)

    assert len(by_group["real-main"]) == 2 * len(DATA.real)
    assert len(by_group["synthetic-cheap"]) == len(DATA.synthetic)
    assert len(by_group["injected"]) == 5
    assert len(by_group["gemini"]) == len(DATA.real)
    assert len({job.key for job in jobs}) == len(jobs)  # every job has its own key
    assert plan_jobs(load_plan(write_plan(tmp_path)), DATA) == jobs  # the same every time


@pytest.mark.parametrize(
    ("group", "message"),
    [
        ({"name": "x", "cases": "all", "model": "anthropic:claude-haiku-5-5"}, "cases"),
        ({"name": "x", "cases": "real"}, "model"),
        (
            {
                "name": "x",
                "cases": "real",
                "model": "anthropic:claude-haiku-5-5",
                "inject": ["meteor"],
            },
            "inject",
        ),
        ({"name": "x", "cases": "real", "model": "nonsense"}, "model"),
    ],
)
def test_a_bad_plan_is_refused(tmp_path: Path, group: dict[str, Any], message: str) -> None:
    from issue_triager.batch import PlanError, load_plan

    with pytest.raises(PlanError, match=message):
        load_plan(write_plan(tmp_path, groups=[group]))


# ---- running ---------------------------------------------------------------------


def fake_triage(
    results: dict[str, Any] | None = None,
) -> tuple[Callable[[Any, int], Any], list[str]]:
    """A triage stand-in: returns an outcome per job, or raises what `results` says."""
    from issue_triager.batch import Outcome

    calls: list[str] = []

    def triage(job: Any, attempt: int = 1) -> Outcome:
        calls.append(job.key)
        planned = (results or {}).get(job.key)
        if isinstance(planned, list) and planned:
            raise planned.pop(0)
        return Outcome(status="applied", grade="wrong" if job.case.id == "3" else "correct",
                       cost_usd=0.01, trace_id=f"trace-{len(calls)}")  # fmt: skip

    return triage, calls


def small_jobs(tmp_path: Path, **group: Any) -> list[Any]:
    from issue_triager.batch import load_plan, plan_jobs

    spec = {"name": "g", "cases": "real", "model": "anthropic:claude-haiku-5-5",
            "sample": 4} | group  # fmt: skip
    return plan_jobs(load_plan(write_plan(tmp_path, groups=[spec])), DATA)


def manifest_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_every_run_is_written_to_the_manifest(tmp_path: Path) -> None:
    from issue_triager.batch import run_batch

    jobs = small_jobs(tmp_path)
    triage, _ = fake_triage()
    manifest = tmp_path / "manifest.jsonl"
    summary = run_batch(jobs, triage, manifest=manifest, budget_usd=10, sleep=lambda s: None)

    lines = manifest_lines(manifest)
    assert [line["key"] for line in lines] == [job.key for job in jobs]
    assert {"group", "case", "repeat", "model", "seed", "status", "grade", "cost_usd",
            "trace_id"} <= set(lines[0])  # fmt: skip
    assert summary.done == 4 and summary.stopped is None
    assert summary.cost_usd == pytest.approx(0.04)


def test_a_batch_resumes_where_it_stopped(tmp_path: Path) -> None:
    from issue_triager.batch import run_batch

    jobs = small_jobs(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    first, _ = fake_triage()
    run_batch(jobs[:2], first, manifest=manifest, budget_usd=10, sleep=lambda s: None)

    again, calls = fake_triage()
    summary = run_batch(jobs, again, manifest=manifest, budget_usd=10, sleep=lambda s: None)

    assert calls == [job.key for job in jobs[2:]]  # the first two aren't run again
    assert summary.skipped == 2 and len(manifest_lines(manifest)) == 4


def test_the_spend_cap_stops_the_batch_cleanly(tmp_path: Path) -> None:
    from issue_triager.batch import run_batch

    jobs = small_jobs(tmp_path)
    triage, calls = fake_triage()
    manifest = tmp_path / "manifest.jsonl"
    summary = run_batch(jobs, triage, manifest=manifest, budget_usd=0.02, sleep=lambda s: None)

    assert len(calls) == 2 and summary.stopped == "budget"
    # spend already in the manifest counts on resume
    again, more = fake_triage()
    assert run_batch(jobs, again, manifest=manifest, budget_usd=0.02,
                     sleep=lambda s: None).stopped == "budget"  # fmt: skip
    assert more == []


GEMINI_429 = (
    "Error calling model 'gemini-3.5-flash-lite' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "{'error': {'code': 429, 'message': 'You exceeded your current quota, please check your "
    "plan and billing details. Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 15, "
    "quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier. Please retry in 21s.'}}"
)
GEMINI_DAILY = GEMINI_429.replace("PerMinute", "PerDay").replace("limit: 15", "limit: 1000")


def test_rate_limits_back_off_and_retry(tmp_path: Path) -> None:
    """Gemini's per-minute 429 mentions billing; it's still a rate limit, not out of credit."""
    from langchain_google_genai.chat_models import GoogleRateLimitError

    from issue_triager.batch import run_batch

    jobs = small_jobs(tmp_path, sample=1)
    errors = [GoogleRateLimitError(GEMINI_429), GoogleRateLimitError(GEMINI_429)]
    triage, calls = fake_triage({jobs[0].key: errors})
    waits: list[float] = []
    run_batch(jobs, triage, manifest=tmp_path / "m.jsonl", budget_usd=10, sleep=waits.append)

    assert len(calls) == 3  # two rate-limited attempts, then success
    assert waits[0] < waits[1]  # backing off
    assert manifest_lines(tmp_path / "m.jsonl")[0]["status"] == "applied"


def test_other_errors_are_recorded_and_the_batch_goes_on(tmp_path: Path) -> None:
    from issue_triager.batch import run_batch

    jobs = small_jobs(tmp_path, sample=2)
    triage, _ = fake_triage({jobs[0].key: [ValueError("bad response")]})
    summary = run_batch(jobs, triage, manifest=tmp_path / "m.jsonl", budget_usd=10,
                        sleep=lambda s: None)  # fmt: skip

    lines = manifest_lines(tmp_path / "m.jsonl")
    assert lines[0]["status"] == "error" and "bad response" in lines[0]["error"]
    assert lines[1]["status"] == "applied" and summary.errors == 1
    # errors are tried again on resume
    again, retried = fake_triage()
    run_batch(jobs, again, manifest=tmp_path / "m.jsonl", budget_usd=10, sleep=lambda s: None)
    assert retried == [jobs[0].key]


def test_paced_groups_pause_between_runs(tmp_path: Path) -> None:
    from issue_triager.batch import run_batch

    jobs = small_jobs(tmp_path, sample=3, pause_seconds=4)
    triage, _ = fake_triage()
    waits: list[float] = []
    run_batch(jobs, triage, manifest=tmp_path / "m.jsonl", budget_usd=10, sleep=waits.append)

    assert waits == [4, 4]  # between runs, not after the last


def test_batch_command_records_each_run_and_its_grade(
    clean_env: pytest.MonkeyPatch,
    tmp_path: Path,
    collector: Collector,  # noqa: F811
) -> None:
    def fake(*args: Any, **kwargs: Any) -> Any:
        return langgraph_models(iter([call("search_issues", {"query": "x"}), decide()]))

    clean_env.setattr(cli, "resolve_models", fake)
    plan = write_plan(tmp_path, groups=[{"name": "g", "cases": "real", "sample": 2,
                                         "model": "openai:gpt-5.4-nano"}])  # fmt: skip
    manifest = tmp_path / "m.jsonl"
    result = CliRunner().invoke(cli.app, ["batch", str(plan), "--manifest", str(manifest),
                                          "--record", collector.url,
                                          "--runs-dir", str(tmp_path)])  # fmt: skip

    assert result.exit_code == 0, result.output
    lines = manifest_lines(manifest)
    roots = [s for s in collector.raw_spans() if not s.parent_span_id]
    assert len(lines) == len(roots) == 2
    assert {line["trace_id"] for line in lines} == {r.trace_id.hex() for r in roots}
    for root in roots:
        attrs = {a.key: a.value.string_value for a in root.attributes}
        assert attrs["triager.grade"] in ("correct", "wrong")
        assert attrs["triager.grade.labels"] in ("ok", "wrong")
        assert attrs["triager.batch.group"] == "g" and attrs["triager.batch.attempt"] == "1"
    assert {line["grade"] for line in lines} <= {"correct", "wrong"}
    assert "done:    2 runs" in result.output


def test_each_attempt_is_its_own_trace_and_the_manifest_names_the_last(
    clean_env: pytest.MonkeyPatch,
    tmp_path: Path,
    collector: Collector,  # noqa: F811
) -> None:
    from langchain_google_genai.chat_models import GoogleRateLimitError

    attempts = iter([GoogleRateLimitError(GEMINI_429), None])
    real_run = cli.run_langgraph

    def flaky(*args: Any, **kwargs: Any) -> Any:
        if (error := next(attempts)) is not None:
            raise error  # inside the root span, as a real model call would
        return real_run(*args, **kwargs)

    def fake(*args: Any, **kwargs: Any) -> Any:
        return langgraph_models(iter([call("search_issues", {"query": "x"}), decide()]))

    clean_env.setattr(cli, "resolve_models", fake)
    clean_env.setattr(cli, "run_langgraph", flaky)
    clean_env.setattr("issue_triager.batch.FIRST_BACKOFF", 0.0)
    gemini = {
        "name": "g",
        "cases": "real",
        "sample": 1,
        "model": "google_genai:gemini-3.5-flash-lite",
    }
    plan = write_plan(tmp_path, groups=[gemini])
    manifest = tmp_path / "m.jsonl"
    args = ["batch", str(plan), "--manifest", str(manifest), "--record", collector.url]
    result = CliRunner().invoke(cli.app, [*args, "--runs-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    roots = [s for s in collector.raw_spans() if not s.parent_span_id]
    by_attempt = {
        next(a.value.string_value for a in r.attributes if a.key == "triager.batch.attempt"): r
        for r in roots
    }
    [line] = manifest_lines(manifest)
    assert set(by_attempt) == {"1", "2"}
    assert line["trace_id"] == by_attempt["2"].trace_id.hex()


def test_batch_exits_nonzero_when_runs_fail_or_a_provider_stops(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("provider unavailable")

    clean_env.setattr(cli, "resolve_models", broken)
    plan = write_plan(tmp_path, groups=[{"name": "g", "cases": "real", "sample": 1,
                                         "model": "openai:gpt-5.4-nano"}])  # fmt: skip
    result = CliRunner().invoke(cli.app, ["batch", str(plan), "--manifest",
                                          str(tmp_path / "m.jsonl")])  # fmt: skip
    assert result.exit_code == 1
    missing = CliRunner().invoke(cli.app, ["batch", str(plan), "--dataset",
                                           str(tmp_path / "nowhere")])  # fmt: skip
    assert missing.exit_code == 2 and "no dataset" in missing.output


def test_batch_dry_run_lists_jobs_without_running(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli.app, ["batch", str(write_plan(tmp_path)), "--dry-run"])

    assert result.exit_code == 0, result.output
    total = 2 * len(DATA.real) + len(DATA.synthetic) + 5 + len(DATA.real)
    assert f"{total} runs" in result.output


def two_providers(tmp_path: Path) -> list[Any]:
    from issue_triager.batch import load_plan, plan_jobs

    groups = [
        {"name": "a", "cases": "real", "sample": 3, "model": "anthropic:claude-haiku-5-5"},
        {"name": "g", "cases": "real", "sample": 2, "model": "google_genai:gemini-3.5-flash-lite"},
    ]
    return plan_jobs(load_plan(write_plan(tmp_path, groups=groups)), DATA)


def anthropic_out_of_credit() -> Exception:
    import httpx
    from langchain_anthropic.chat_models import AnthropicInvalidRequestError

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request)
    body = {"type": "error", "error": {"type": "invalid_request_error",
            "message": "Your credit balance is too low to access the Anthropic API."}}  # fmt: skip
    return AnthropicInvalidRequestError(
        "Error code: 400 - " + str(body),
        response=response,  # pyright: ignore[reportArgumentType]  # SDK types a newer httpx
        body=body,
    )


def test_a_billing_error_stops_only_that_provider(tmp_path: Path) -> None:
    """Out of credit, every later run on that provider would fail too; the others go on."""
    from issue_triager.batch import run_batch

    jobs = two_providers(tmp_path)
    triage, calls = fake_triage({jobs[0].key: [anthropic_out_of_credit()]})
    summary = run_batch(jobs, triage, manifest=tmp_path / "m.jsonl", budget_usd=10,
                        sleep=lambda s: None)  # fmt: skip

    assert calls == [jobs[0].key, jobs[3].key, jobs[4].key]  # Anthropic stops, Gemini runs
    assert summary.provider_stops == {"anthropic": "billing"}
    assert manifest_lines(tmp_path / "m.jsonl")[0]["status"] == "error"


def test_a_spent_daily_quota_stops_only_that_provider(tmp_path: Path) -> None:
    from langchain_google_genai.chat_models import GoogleRateLimitError

    from issue_triager.batch import run_batch

    jobs = two_providers(tmp_path)
    triage, calls = fake_triage({jobs[3].key: [GoogleRateLimitError(GEMINI_DAILY)]})
    waits: list[float] = []
    summary = run_batch(jobs, triage, manifest=tmp_path / "m.jsonl", budget_usd=10,
                        sleep=waits.append)  # fmt: skip

    assert calls == [j.key for j in jobs[:4]] and waits == []  # no hopeless backoff
    assert summary.provider_stops == {"google_genai": "daily quota"}


def test_overloaded_is_retried_but_other_errors_with_429_in_them_are_not(
    tmp_path: Path,
) -> None:
    import httpx
    from langchain_anthropic.chat_models import AnthropicOverloadedError

    from issue_triager.batch import run_batch

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    overloaded = AnthropicOverloadedError(
        "Overloaded",
        response=httpx.Response(529, request=request),  # pyright: ignore[reportArgumentType]
        body=None,
    )
    jobs = small_jobs(tmp_path, sample=2)
    triage, calls = fake_triage({
        jobs[0].key: [overloaded],
        jobs[1].key: [ValueError("prompt is 204291 tokens, too long")],
    })  # fmt: skip
    run_batch(jobs, triage, manifest=tmp_path / "m.jsonl", budget_usd=10, sleep=lambda s: None)

    assert calls == [jobs[0].key, jobs[0].key, jobs[1].key]
    assert [line["status"] for line in manifest_lines(tmp_path / "m.jsonl")] == ["applied", "error"]


def test_missing_credentials_stop_that_provider(tmp_path: Path) -> None:
    from flight_recorder.llm import MissingCredentialsError

    from issue_triager.batch import run_batch

    jobs = two_providers(tmp_path)
    no_key = MissingCredentialsError("google_genai needs GOOGLE_API_KEY")
    triage, calls = fake_triage({jobs[3].key: [no_key]})
    summary = run_batch(jobs, triage, manifest=tmp_path / "m.jsonl", budget_usd=10,
                        sleep=lambda s: None)  # fmt: skip

    assert calls == [j.key for j in jobs[:4]]
    assert summary.provider_stops == {"google_genai": "credentials"}


def test_a_truncated_last_manifest_line_is_ignored(tmp_path: Path) -> None:
    from issue_triager.batch import run_batch

    jobs = small_jobs(tmp_path, sample=2)
    manifest = tmp_path / "m.jsonl"
    first, _ = fake_triage()
    run_batch(jobs[:1], first, manifest=manifest, budget_usd=10, sleep=lambda s: None)
    with manifest.open("a", encoding="utf-8") as file:
        file.write('{"key": "g:anthropic')  # a crash mid-write

    again, calls = fake_triage()
    run_batch(jobs, again, manifest=manifest, budget_usd=10, sleep=lambda s: None)
    assert calls == [jobs[1].key]


def test_the_manifest_key_includes_the_model(tmp_path: Path) -> None:
    """Changing a group's model makes new runs, not 'already done'."""
    haiku = small_jobs(tmp_path, sample=1)
    sonnet = small_jobs(tmp_path, sample=1, model="anthropic:claude-sonnet-5-5")
    assert haiku[0].key != sonnet[0].key and "claude-haiku-5-5" in haiku[0].key


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"groups": [{"name": "x", "cases": "real", "model": "anthropic:claude-unknown-1"}]},
         "price"),
        ({"groups": [{"name": "x", "cases": "real", "model": "anthropic:claude-haiku-5-5",
                      "rate": 1.5}]}, "rate"),
        ({"budget_usd": float("nan")}, "budget"),
    ],
)  # fmt: skip
def test_plans_that_would_break_the_cap_are_refused(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    from issue_triager.batch import PlanError, load_plan

    with pytest.raises(PlanError, match=message):
        load_plan(write_plan(tmp_path, **overrides))


def test_the_corpus_plan_is_the_agreed_mix() -> None:
    from issue_triager.batch import load_plan, plan_jobs

    plan = load_plan(Path(__file__).parents[1] / "batches" / "afr-corpus.yaml")
    counts: dict[str, int] = {}
    for job in plan_jobs(plan, DATA):
        counts[job.group.name] = counts.get(job.group.name, 0) + 1
    real, synthetic = len(DATA.real), len(DATA.synthetic)
    assert counts == {"real-mini": 2 * real, "synthetic-nano": synthetic,
                      "injected-mini": 30, "real-gemini": real}  # fmt: skip
    models = {g.name: (g.model, g.comment_model) for g in plan.groups}
    assert models["real-mini"] == ("openai:gpt-5.4-mini", "openai:gpt-5.4-nano")
    assert models["synthetic-nano"][0] == "openai:gpt-5.4-nano"
    assert models["injected-mini"] == ("openai:gpt-5.4-mini", "openai:gpt-5.4-nano")
    assert models["real-gemini"][0] == "google_genai:gemini-3.5-flash-lite"  # unchanged: done
    assert plan.budget_usd == 10


def test_a_rejected_run_is_still_graded() -> None:
    from issue_triager.grading import grade_attributes

    truth = REAL_19.truth
    right = {"labels": list(truth.labels), "milestone": truth.milestone,
             "duplicate_of": truth.duplicate_of, "regression_of": truth.regression_of}  # fmt: skip
    assert (
        grade_attributes({"status": "rejected", "decision": right}, REAL_19)["triager.grade"]
        == "correct"
    )


def test_a_run_that_errors_still_records_its_trace_id(
    clean_env: pytest.MonkeyPatch,
    tmp_path: Path,
    collector: Collector,  # noqa: F811
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("provider unavailable")

    clean_env.setattr(cli, "resolve_models", lambda *a, **k: langgraph_models(iter([decide()])))
    clean_env.setattr(cli, "run_langgraph", broken)
    plan = write_plan(tmp_path, groups=[{"name": "g", "cases": "real", "sample": 1,
                                         "model": "openai:gpt-5.4-nano"}])  # fmt: skip
    manifest = tmp_path / "m.jsonl"
    args = ["batch", str(plan), "--manifest", str(manifest), "--record", collector.url]
    CliRunner().invoke(cli.app, [*args, "--runs-dir", str(tmp_path)])

    [line] = manifest_lines(manifest)
    [root] = [s for s in collector.raw_spans() if not s.parent_span_id]
    assert line["status"] == "error" and line["trace_id"] == root.trace_id.hex()


def test_a_budget_given_on_the_command_line_is_checked_like_the_plans(tmp_path: Path) -> None:
    plan = write_plan(tmp_path, budget_usd=None,
                      groups=[{"name": "x", "cases": "real",
                               "model": "anthropic:claude-unknown-1"}])  # fmt: skip
    priced = CliRunner().invoke(cli.app, ["batch", str(plan), "--budget", "5", "--dry-run"])
    assert priced.exit_code == 2 and "price" in priced.output
    nan = CliRunner().invoke(cli.app, ["batch", str(write_plan(tmp_path)), "--budget", "nan"])
    assert nan.exit_code == 2 and "budget" in nan.output


def test_a_503_is_retried_and_openai_out_of_quota_stops_the_provider() -> None:
    from issue_triager.batch import classify

    class Unavailable(Exception):
        status_code = 503

    class OpenAIRateLimitError(Exception):
        status_code = 429

    assert classify(Unavailable("The model is overloaded."))[0] == "retry"
    quota = OpenAIRateLimitError("Error code: 429 - {'error': {'code': 'insufficient_quota'}}")
    assert classify(quota) == ("stop", "billing")


def test_a_real_google_503_is_retried() -> None:
    """The real GoogleAPIError keeps its status in `code`, not `status_code`."""
    from langchain_google_genai.chat_models import GoogleAPIError

    from issue_triager.batch import classify

    body = {"error": {"code": 503, "status": "UNAVAILABLE",
                      "message": "This model is currently experiencing high demand."}}  # fmt: skip
    error = GoogleAPIError(code=503, response_json=body, response=None)
    assert classify(error)[0] == "retry"
