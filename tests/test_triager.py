import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typer.testing import CliRunner

from issue_triager import cli
from issue_triager.dataset import RepoInfo, Rules, load_dataset
from issue_triager.decision import decision_model
from issue_triager.graph import Models, TriageState, build_graph
from issue_triager.prompts import triage_system
from issue_triager.search import KeywordIndex
from issue_triager.tools import FailureInjector, make_tools, status_of

DATA = load_dataset()
REAL = list(DATA.corpus.values())
CASES = DATA.by_id
CORPUS = DATA.corpus
# An issue that's still open in the bundled data. Found, not hard-coded: re-exports close them.
STILL_OPEN = next(
    c.id
    for c in DATA.corpus.values()
    if c.state != "closed" and not c.truth.duplicate_of and c.id != "19"  # 19: the triaged one
)


class FakeModel(GenericFakeChatModel):
    """Replays scripted messages; ignores tool binding like a model that just complies."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeModel":  # type: ignore[override]
        return self


def scripted(*messages: AIMessage) -> FakeModel:
    return FakeModel(messages=iter(messages))


def call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}]
    )


def decide(**overrides: Any) -> AIMessage:
    args = {
        "labels": ["evals", "infra"],
        "milestone": "Week 3: Evals and guardrails",
        "duplicate_of": None,
        "regression_of": None,
        "confidence": 0.8,
        "rationale": "CI for evals.",
    } | overrides
    return call("TriageDecision", args)


def models(
    triage: Iterator[AIMessage] | FakeModel, comment: str = "Looks like evals work."
) -> Models:
    return Models(
        triage=triage if isinstance(triage, FakeModel) else FakeModel(messages=triage),
        comment=scripted(AIMessage(content=comment)),
        triage_spec="openai:gpt-5.4-nano",
        comment_spec="openai:gpt-5.4-nano",
    )


def run(case_id: str, m: Models, tmp_path: Path, *, approve: bool) -> dict[str, Any]:
    graph = build_graph(CASES[case_id], DATA, m, checkpointer=InMemorySaver(), runs_dir=tmp_path)
    config: RunnableConfig = {"configurable": {"thread_id": "t"}}
    state = graph.invoke(cast(TriageState, {"case_id": case_id}), config)
    if "__interrupt__" in state:
        state = graph.invoke(Command(resume=approve), config)
    return dict(state)


# ---- Search and tools ------------------------------------------------------------


def test_keyword_search_finds_the_obvious_match() -> None:
    hits = KeywordIndex(REAL).search("provider switchable LLM config")

    assert hits[0][0].id in {"2", "3"}


def test_search_never_returns_the_target() -> None:
    hits = KeywordIndex(REAL).search("provider switchable LLM config", exclude="2")

    assert "2" not in {case.id for case, _ in hits}


def test_tools_hide_the_issue_being_triaged() -> None:
    search, get = make_tools(REAL, target_id="19")

    found = json.loads(search.invoke({"query": "run evals in CI"}))

    assert "19" not in {item["id"] for item in found}
    assert "not found" in get.invoke({"issue_id": "19"})


def test_status_distinguishes_done_open_and_duplicate() -> None:
    assert status_of(CORPUS["1"]) == "done (closed)"
    assert status_of(CORPUS["3"]) == "closed as duplicate of #2"
    assert status_of(CORPUS[STILL_OPEN]) == "open"


@pytest.mark.parametrize(
    ("mode", "check"),
    [
        ("timeout", lambda out: "timed out" in out),
        ("empty", lambda out: out == "[]"),
        ("bad-data", lambda out: all(i["labels"] == [] for i in json.loads(out))),
        ("noise", lambda out: len(json.loads(out)) == 5),
    ],
)
def test_injected_failures_reach_the_agent(mode: Any, check: Any) -> None:
    injector = FailureInjector([mode], rate=1.0, seed=1)
    search, _ = make_tools(REAL, target_id="19", injector=injector)

    out = search.invoke({"query": "evals CI"})

    assert check(out)
    assert injector.injected == [f"search_issues:{mode}"]


def test_no_injection_by_default() -> None:
    injector = FailureInjector()
    search, _ = make_tools(REAL, target_id="19", injector=injector)

    search.invoke({"query": "evals"})

    assert injector.injected == []


# ---- Per-repo schema and prompt -------------------------------------------------


def test_decision_schema_only_allows_the_repos_labels_and_milestones() -> None:
    schema = decision_model(DATA.repo)
    ok = {"labels": ["ui"], "milestone": None, "confidence": 1, "rationale": "-"}

    schema.model_validate(ok)
    with pytest.raises(ValueError):
        schema.model_validate(ok | {"labels": ["wontfix"]})
    with pytest.raises(ValueError):
        schema.model_validate(ok | {"milestone": "Week 9"})


def test_schema_for_a_repo_without_milestones() -> None:
    schema = decision_model(RepoInfo(repo="a/b", labels=[]))

    decision = schema.model_validate({"labels": ["anything"], "confidence": 1, "rationale": "-"})

    assert decision.milestone is None


def test_prompt_is_built_from_the_repo() -> None:
    prompt = triage_system(DATA.repo, DATA.rules)

    assert "sonicjoy/agent-flight-recorder: Record, replay" in prompt
    assert "- recorder: Ingest, storage, redaction, search, CLI" in prompt
    assert "Week 3: Evals and guardrails" in prompt
    assert "These labels need a milestone: stretch." in prompt


def test_prompt_skips_empty_sections() -> None:
    prompt = triage_system(RepoInfo(repo="a/b"), Rules())

    assert "Labels:" not in prompt
    assert "Milestones" not in prompt
    assert "Repo rules" not in prompt


# ---- Graph -----------------------------------------------------------------------


def test_approved_run_searches_decides_and_writes_dry_run_file(tmp_path: Path) -> None:
    m = models(iter([call("search_issues", {"query": "evals CI"}), decide()]))

    state = run("19", m, tmp_path, approve=True)

    assert state["status"] == "applied"
    record = json.loads(Path(state["output_path"]).read_text(encoding="utf-8"))
    assert record["decision"]["labels"] == ["evals", "infra"]
    assert record["comment"] == "Looks like evals work."
    assert record["steps"][0]["tool"] == "search_issues"
    assert record["steps"][0]["status"] == "success"


def test_rejected_run_writes_nothing(tmp_path: Path) -> None:
    state = run("19", models(iter([decide()])), tmp_path, approve=False)

    assert state["status"] == "rejected"
    assert list(tmp_path.iterdir()) == []


def test_invalid_decision_is_retried_with_feedback(tmp_path: Path) -> None:
    m = models(iter([decide(labels=["ui", "stretch"], milestone=None), decide()]))

    state = run("19", m, tmp_path, approve=True)

    assert state["attempts"] == 2
    assert state["status"] == "applied"


def test_gives_up_after_two_invalid_decisions(tmp_path: Path) -> None:
    m = models(iter([decide(regression_of=STILL_OPEN), decide(regression_of=STILL_OPEN)]))

    state = run("19", m, tmp_path, approve=True)

    assert state["status"] == "failed"
    assert "isn't done" in state["feedback"][0]
    assert list(tmp_path.iterdir()) == []


# ---- CLI -------------------------------------------------------------------------

runner = CliRunner()


def test_cli_rejects_unknown_issue() -> None:
    result = runner.invoke(cli.app, ["run", "--issue", "nope"])

    assert result.exit_code == 2
    assert "no issue" in result.output


def test_cli_rejects_unknown_failure_mode() -> None:
    result = runner.invoke(cli.app, ["run", "--issue", "19", "--inject", "meteor"])

    assert result.exit_code == 2


def test_cli_explains_missing_key(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clean_env.chdir(tmp_path)  # no .env here

    result = runner.invoke(cli.app, ["run", "--issue", "19", "--model", "openai:gpt-5.4-nano"])

    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in result.output


def test_cli_end_to_end_with_auto_approve(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = models(iter([call("search_issues", {"query": "evals"}), decide()]))
    monkeypatch.setattr(cli, "resolve_models", lambda *a, **k: fake)

    result = runner.invoke(cli.app, ["run", "--issue", "19", "--yes", "--runs-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "vs truth: labels ok; milestone ok; duplicate_of ok" in result.output
    assert "status:  applied" in result.output
    assert "cost: $" in result.output
