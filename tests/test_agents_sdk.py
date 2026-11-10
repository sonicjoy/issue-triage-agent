"""The OpenAI Agents SDK version, driven by the SDK's own scripted model (no network)."""

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from agents import FunctionTool
from agents.testing import ScriptedModel, assistant_message, function_call
from agents.usage import Usage
from typer.testing import CliRunner

from issue_triager import agents_sdk, cli
from issue_triager.agents_sdk import MAX_TOOL_CALLS, MAX_TURNS, SdkModels
from issue_triager.dataset import load_dataset
from issue_triager.tools import FailureInjector

DATA = load_dataset()
CASE = DATA.by_id["19"]
GOOD = {
    "labels": ["evals", "infra"],
    "milestone": "Week 3: Evals and guardrails",
    "duplicate_of": None,
    "regression_of": None,
    "confidence": 0.8,
    "rationale": "CI for evals.",
}
NO_LABELS = GOOD | {"labels": []}


def call(name: str, args: dict[str, Any]) -> list[Any]:
    return [function_call(name, args, call_id=f"call_{uuid.uuid4().hex[:8]}")]


def search(query: str = "eval CI") -> list[Any]:
    return call("search_issues", {"query": query})


def apply(decision: dict[str, Any] = GOOD) -> list[Any]:
    return call("apply_triage", decision)


def models(*steps: list[Any], comment: str = "Looks like evals work.") -> SdkModels:
    usage = Usage(requests=1, input_tokens=100, output_tokens=10, total_tokens=110)
    return SdkModels(
        triage=ScriptedModel(steps, default_usage=usage),
        comment=ScriptedModel([[assistant_message(comment)]] * 2, default_usage=usage),
        triage_spec="openai:gpt-5.4-nano",
        comment_spec="openai:gpt-5.4-nano",
    )


class Reviewer:
    """Records what the human was shown and answers with `approve`."""

    def __init__(self, approve: bool) -> None:
        self.approve = approve
        self.seen: list[dict[str, Any]] = []

    def __call__(self, state: dict[str, Any]) -> bool:
        self.seen.append(json.loads(json.dumps(state)))
        return self.approve


def run(
    m: SdkModels, tmp_path: Path, *, approve: bool = True, injector: FailureInjector | None = None
) -> tuple[dict[str, Any], Reviewer]:
    reviewer = Reviewer(approve)
    state = agents_sdk.triage(CASE, DATA, m, review=reviewer, injector=injector, runs_dir=tmp_path)
    return state, reviewer


def tool_outputs(model: Any) -> list[str]:
    """What the model was told by tools, from the input of its last call."""
    return [
        str(item.get("output"))
        for item in model.last_call.input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]


def test_approved_run_searches_decides_and_writes_dry_run_file(tmp_path: Path) -> None:
    m = models(search(), apply())
    state, reviewer = run(m, tmp_path)

    assert state["status"] == "applied"
    assert [s["tool"] for s in state["steps"]] == ["search_issues", "apply_triage"]
    assert reviewer.seen[0]["decision"] == GOOD
    assert reviewer.seen[0]["comment"] == "Looks like evals work."
    record = json.loads(Path(state["output_path"]).read_text(encoding="utf-8"))
    assert record["engine"] == "openai-agents"
    assert record["decision"]["labels"] == ["evals", "infra"]
    assert record["comment"] == "Looks like evals work."
    assert {u["step"] for u in state["usage"]} == {"triage", "comment"}


def test_rejected_run_writes_nothing(tmp_path: Path) -> None:
    state, reviewer = run(models(apply()), tmp_path, approve=False)

    assert state["status"] == "rejected"
    assert len(reviewer.seen) == 1
    assert not list(tmp_path.glob("triage-*.json"))


def test_invalid_decision_goes_back_to_the_model_not_the_human(tmp_path: Path) -> None:
    m = models(apply(NO_LABELS), apply())
    state, reviewer = run(m, tmp_path)

    assert state["status"] == "applied"
    assert state["attempts"] == 2
    assert len(reviewer.seen) == 1  # only the valid decision reached the human
    assert reviewer.seen[0]["decision"] == GOOD
    assert any("broke these rules" in out for out in tool_outputs(m.triage))


def test_gives_up_after_two_invalid_decisions(tmp_path: Path) -> None:
    state, reviewer = run(models(apply(NO_LABELS), apply(NO_LABELS)), tmp_path)

    assert state["status"] == "failed"
    assert state["attempts"] == 2
    assert state["feedback"]
    assert reviewer.seen == []
    assert not list(tmp_path.glob("triage-*.json"))


def test_tool_errors_reach_the_model(tmp_path: Path) -> None:
    m = models(search(), apply())
    injector = FailureInjector(["timeout"], rate=1.0)
    state, _ = run(m, tmp_path, injector=injector)

    assert state["steps"][0]["status"] == "error"
    assert "search_issues timed out after 30s" in tool_outputs(m.triage)


def test_tool_calls_are_capped(tmp_path: Path) -> None:
    m = models(*[search() for _ in range(MAX_TOOL_CALLS + 1)], apply())
    state, _ = run(m, tmp_path)

    assert state["status"] == "applied"
    statuses = [s["status"] for s in state["steps"] if s["tool"] == "search_issues"]
    assert statuses == ["success"] * MAX_TOOL_CALLS + ["error"]
    assert any("Tool call limit" in out for out in tool_outputs(m.triage))


def test_run_without_a_decision_fails_at_the_turn_limit(tmp_path: Path) -> None:
    state, reviewer = run(models(*[search() for _ in range(MAX_TURNS)]), tmp_path)

    assert state["status"] == "failed"
    assert reviewer.seen == []


def test_apply_tool_takes_the_repos_own_schema(tmp_path: Path) -> None:
    m = models(apply())
    run(m, tmp_path)

    assert isinstance(m.triage, ScriptedModel)
    first = m.triage.first_call
    assert first is not None
    tools = {t.name: t for t in first.tools}
    assert set(tools) == {"search_issues", "get_issue", "apply_triage"}
    apply_tool = tools["apply_triage"]
    assert isinstance(apply_tool, FunctionTool)
    labels = apply_tool.params_json_schema["properties"]["labels"]["items"]
    assert set(labels["enum"]) == DATA.repo.label_names
    assert first.model_settings.tool_choice == "required"


def test_trace_is_written_locally_and_not_uploaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uploader = agents_sdk.default_processor()  # the SDK's upload to OpenAI

    def no_upload(*args: Any) -> None:
        raise AssertionError("the OpenAI trace exporter must stay off by default")

    for hook in ("on_trace_start", "on_trace_end", "on_span_start", "on_span_end"):
        monkeypatch.setattr(uploader, hook, no_upload)
    run(models(search(), apply()), tmp_path)

    assert uploader not in agents_sdk._registered_processors()  # pyright: ignore[reportPrivateUsage]

    lines = [
        json.loads(line)
        for path in tmp_path.glob("trace-*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    span_types = {line["span_data"]["type"] for line in lines if line["event"] == "span"}
    assert {"agent", "function"} <= span_types
    assert lines[0]["event"] == "trace_start"


def test_cli_rejects_unknown_engine() -> None:
    result = CliRunner().invoke(cli.app, ["run", "--issue", "19", "--engine", "crewai"])
    assert result.exit_code == 2
    assert "unknown engine" in result.output


def test_cli_end_to_end_with_auto_approve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    m = models(search(), apply())
    monkeypatch.setattr(cli, "resolve_sdk_models", lambda *a, **k: m)
    result = CliRunner().invoke(
        cli.app,
        ["run", "--issue", "19", "--engine", "openai-agents", "--yes", "--runs-dir", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert "engine:  openai-agents" in result.output
    assert "status:  applied" in result.output
    assert "tokens:  300 in / 30 out" in result.output
