"""The eval suite's adapter, dataset hash and generator (AFR #16)."""

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml
from fakes import FakeModel, decide, scripted
from flight_recorder.evals import MARK
from flight_recorder.evals.cases import load
from flight_recorder.evals.checks import canonical, run_checks
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from issue_triager import cli
from issue_triager import eval as suite
from issue_triager.dataset import DEFAULT_DATASET, GroundTruth, TriageCase, load_dataset
from issue_triager.tools import summarise

ROOT = Path(__file__).parents[1]
DATA = load_dataset()
SPEC = "openai:gpt-5.4-mini"


def copy_dataset(tmp_path: Path) -> Path:
    target = tmp_path / "dataset"
    shutil.copytree(DEFAULT_DATASET, target)
    return target


def edit_json(path: Path, change: Any) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    change(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")


def edit_yaml(path: Path, change: Any) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(raw)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")


def issue(raw: dict[str, Any], issue_id: str) -> dict[str, Any]:
    return next(i for i in raw["issues"] if i["id"] == issue_id)


# ---- dataset_hash ----------------------------------------------------------------


def test_the_hash_is_summarise_repo_and_rules_as_specified() -> None:
    corpus = [summarise(c, full=True) for c in sorted(DATA.corpus.values(), key=lambda c: c.id)]
    repo = json.loads((DEFAULT_DATASET / "repo.json").read_text(encoding="utf-8"))
    rules = yaml.safe_load((DEFAULT_DATASET / "rules.yaml").read_text(encoding="utf-8"))
    text = canonical({"corpus": corpus, "repo": repo, "rules": rules})

    assert suite.dataset_hash(DATA) == hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@pytest.mark.parametrize(
    "change",
    [
        lambda raw: issue(raw, "5").update(body="Changed."),
        lambda raw: issue(raw, "5")["truth"].update(labels=["ui"]),
        lambda raw: issue(raw, "5")["truth"].update(milestone=None),
        lambda raw: issue(raw, "5")["truth"].update(duplicate_of="2"),
    ],
    ids=["body", "labels", "milestone", "duplicate"],
)
def test_the_hash_changes_with_what_search_shows(tmp_path: Path, change: Any) -> None:
    folder = copy_dataset(tmp_path)
    edit_json(folder / "issues.json", change)

    assert suite.dataset_hash(load_dataset(folder)) != suite.dataset_hash(DATA)


def test_the_hash_changes_with_the_labels_and_the_rules(tmp_path: Path) -> None:
    labels = copy_dataset(tmp_path / "a")
    edit_json(labels / "repo.json", lambda raw: raw["labels"].append({"name": "new"}))
    rules = copy_dataset(tmp_path / "b")
    edit_yaml(rules / "rules.yaml", lambda raw: raw.update(require_label=False))

    assert suite.dataset_hash(load_dataset(labels)) != suite.dataset_hash(DATA)
    assert suite.dataset_hash(load_dataset(rules)) != suite.dataset_hash(DATA)


def test_the_hash_ignores_what_the_agent_cant_see(tmp_path: Path) -> None:
    folder = copy_dataset(tmp_path)
    edit_yaml(folder / "accepted_answers.yaml", lambda raw: raw["accept"].pop("1"))
    edit_yaml(folder / "synthetic.yaml", lambda raw: raw["cases"][0].update(title="Changed."))

    assert suite.dataset_hash(load_dataset(folder)) == suite.dataset_hash(DATA)


def test_the_hash_ignores_line_endings(tmp_path: Path) -> None:
    folder = copy_dataset(tmp_path)
    for path in folder.iterdir():
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))

    assert suite.dataset_hash(load_dataset(folder)) == suite.dataset_hash(DATA)


def test_a_missing_rules_file_hashes_as_null(tmp_path: Path) -> None:
    folder = copy_dataset(tmp_path)
    (folder / "rules.yaml").unlink()
    data = load_dataset(folder)
    corpus = [summarise(c, full=True) for c in sorted(data.corpus.values(), key=lambda c: c.id)]
    repo = json.loads((folder / "repo.json").read_text(encoding="utf-8"))
    text = canonical({"corpus": corpus, "repo": repo, "rules": None})

    assert suite.dataset_hash(data) == hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# ---- triage, the adapter ---------------------------------------------------------


class Status(Exception):
    """A provider error carrying an HTTP status, as the OpenAI and Anthropic SDKs do."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class Failing(FakeModel):
    """A model whose every call fails with `error`."""

    error: Any = None

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise self.error

    def _stream(self, *args: Any, **kwargs: Any) -> Any:
        raise self.error


def comment() -> FakeModel:
    return scripted(AIMessage(content="Looks right."))


def given(case_id: str = "19", **changes: Any) -> dict[str, Any]:
    return suite.case_input(DATA.by_id[case_id], SPEC, suite.dataset_hash(DATA)) | changes


@dataclass
class Calls:
    models: list[Any] = field(default_factory=list[Any])
    """What `chat_model` returns, in call order: triage then comment, per attempt."""
    specs: list[str] = field(default_factory=list[str])
    waits: list[float] = field(default_factory=list[float])


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Calls:
    """Fake models instead of providers, recorded waits instead of sleeps, and an empty
    working folder."""
    record = Calls()

    def chat_model(spec: str, **_: Any) -> Any:
        record.specs.append(spec)
        return record.models.pop(0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(suite, "chat_model", chat_model)
    monkeypatch.setattr(suite, "SLEEP", record.waits.append)
    return record


def test_triage_is_marked_as_an_agent() -> None:
    assert getattr(suite.triage, MARK) is True


def test_triage_returns_the_normalised_fields_and_meta(
    calls: Calls, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls.models += [
        scripted(decide(labels=["infra", "evals"], duplicate_of="#5")),
        comment(),
    ]
    made: list[Path] = []
    real = suite.tempfile.TemporaryDirectory

    def recorded(*args: Any, **kwargs: Any) -> Any:
        folder = real(*args, **kwargs)
        made.append(Path(folder.name))
        return folder

    monkeypatch.setattr(suite.tempfile, "TemporaryDirectory", recorded)

    output = suite.triage(given())

    assert output == {
        "labels": ["evals", "infra"],
        "milestone": "Week 3: Evals and guardrails",
        "duplicate_of": "5",
        "regression_of": None,
        "meta": {"model": SPEC, "status": "applied", "feedback": []},
    }
    assert calls.specs == [SPEC, SPEC]
    assert list(tmp_path.iterdir()) == []
    assert len(made) == 1 and not made[0].exists()  # the dry-run folder is gone


def test_no_decision_fails_every_check(calls: Calls) -> None:
    calls.models += [
        scripted(AIMessage(content="Not sure."), AIMessage(content="Still not sure.")),
        comment(),
    ]
    case = DATA.by_id["s50"]  # truth: milestone, duplicate_of and regression_of all empty

    output = suite.triage(given("s50"))

    assert output["meta"]["status"] == "failed"
    assert {output[name] for name in ("labels", "milestone", "duplicate_of", "regression_of")} == {
        suite.NO_DECISION
    }
    assert not any(r.passed for r in run_checks(suite.checks_for(case), output))


def test_a_failed_validation_is_an_answer(calls: Calls) -> None:
    bad = decide(labels=["stretch"], milestone=None)
    calls.models += [scripted(bad, decide(labels=["stretch"], milestone=None)), comment()]

    output = suite.triage(given())

    assert output["labels"] == ["stretch"]
    assert output["milestone"] is None
    assert output["meta"]["status"] == "failed"
    assert "need a milestone" in output["meta"]["feedback"][0]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"dataset": "000000000000"}, "dataset"),
        ({"id": "999"}, "999"),
        ({"title": "Another title"}, "title"),
        ({"body": "Another body"}, "body"),
        ({"model": None}, "model"),
    ],
    ids=["hash", "unknown-id", "title", "body", "no-model"],
)
def test_triage_refuses_a_case_written_for_another_dataset(
    calls: Calls, changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        suite.triage(given(**changes))
    assert calls.models == []  # refused before any model was built


def test_a_transient_error_is_retried(calls: Calls) -> None:
    calls.models += [Failing(messages=iter([]), error=Status(503)), comment()]
    calls.models += [scripted(decide()), comment()]

    output = suite.triage(given())

    assert output["meta"]["status"] == "applied"
    assert calls.waits == [15]


def test_retries_stop_after_two(calls: Calls) -> None:
    for _ in range(3):
        calls.models += [Failing(messages=iter([]), error=Status(529)), comment()]

    with pytest.raises(Status):
        suite.triage(given())
    assert calls.waits == [15, 30]


def test_other_errors_are_not_retried(calls: Calls) -> None:
    calls.models += [Failing(messages=iter([]), error=Status(401)), comment()]

    with pytest.raises(Status):
        suite.triage(given())
    assert calls.waits == []


# ---- the generator ---------------------------------------------------------------


def selection(*ids: str, model: str = SPEC) -> suite.Selection:
    return suite.Selection(
        model=model, cases=[suite.Selected(id=i, tags=["t"], note="n") for i in ids]
    )


def checks(case_id: str) -> dict[str, dict[str, Any]]:
    case = load(suite.build_suite(DATA, selection(case_id))[f"{case_id}.yaml"])
    return {c["field"]: c for c in case["checks"]}


def test_accepted_answers_become_one_of_with_the_truth_first() -> None:
    assert checks("1")["labels"] == {
        "type": "one_of",
        "field": "labels",
        "values": [["infra"], ["documentation", "infra"]],
    }
    assert checks("s58")["duplicate_of"] == {
        "type": "one_of",
        "field": "duplicate_of",
        "values": ["23", None],
    }


def test_fields_without_accepted_answers_are_equals_with_null_for_nothing() -> None:
    found = checks("3")

    assert found["duplicate_of"] == {"type": "equals", "field": "duplicate_of", "value": "2"}
    assert found["regression_of"] == {"type": "equals", "field": "regression_of", "value": None}
    assert checks("s50")["milestone"] == {"type": "equals", "field": "milestone", "value": None}


def test_an_accepted_answer_equal_to_the_truth_is_listed_once() -> None:
    case = TriageCase(
        id="x",
        title="t",
        body="b",
        source="synthetic",
        truth=GroundTruth.model_validate(
            {
                "labels": ["infra"],
                "milestone": None,
                "accept": {"labels": [["infra"], ["evals"]], "milestone": [None]},
            }
        ),
    )

    found = {c["field"]: c for c in suite.checks_for(case)}

    assert found["labels"]["values"] == [["infra"], ["evals"]]
    assert found["milestone"] == {"type": "equals", "field": "milestone", "value": None}


def test_a_case_file_holds_the_input_the_adapter_checks() -> None:
    files = suite.build_suite(DATA, selection("s47", "3"))
    case = load(files["s47.yaml"])

    assert case["input"] == given("s47")
    assert (case["id"], case["tags"], case["note"]) == ("s47", ["t"], "n")
    assert load(files["suite.yaml"]) == {"agent": "issue_triager.eval:triage", "model": SPEC}


def test_every_selected_input_round_trips_through_a_case_file() -> None:
    chosen = suite.load_selection(ROOT / "evals" / "selection.yaml")
    files = suite.build_suite(DATA, chosen)

    for selected in chosen.cases:
        case = load(files[f"{selected.id}.yaml"])
        assert case["input"] == suite.case_input(
            DATA.by_id[selected.id], chosen.model, suite.dataset_hash(DATA)
        )


def test_an_unknown_id_is_refused() -> None:
    with pytest.raises(ValueError, match="999"):
        suite.build_suite(DATA, selection("3", "999"))


def test_the_selection_refuses_a_duplicate_id_and_reads_numbers_as_text(tmp_path: Path) -> None:
    path = tmp_path / "selection.yaml"
    path.write_text(f"model: {SPEC}\ncases:\n- {{id: 3, tags: [a], note: n}}\n", encoding="utf-8")
    assert suite.load_selection(path).cases[0].id == "3"

    path.write_text(
        f"model: {SPEC}\ncases:\n- {{id: 3, tags: [], note: n}}\n"
        "- {id: '3', tags: [], note: n}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="3"):
        suite.load_selection(path)


# ---- triager suite, the command --------------------------------------------------

runner = CliRunner()


def test_the_command_writes_the_suite_and_keeps_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals" / "cases" / "triager").mkdir(parents=True)
    folder = tmp_path / "evals" / "cases" / "triager"
    (folder / "old.yaml").write_text("id: old\n", encoding="utf-8")
    (folder / "outputs.jsonl").write_text("kept\n", encoding="utf-8")
    (folder / "baseline.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "evals" / "selection.yaml").write_text(
        f"model: {SPEC}\ncases:\n- {{id: '3', tags: [a], note: n}}\n", encoding="utf-8"
    )

    result = runner.invoke(cli.app, ["suite"])

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in folder.iterdir()) == [
        "3.yaml",
        "baseline.json",
        "outputs.jsonl",
        "suite.yaml",
    ]
    assert (folder / "outputs.jsonl").read_text(encoding="utf-8") == "kept\n"
    assert (folder / "baseline.json").read_text(encoding="utf-8") == "{}\n"
    assert b"\r\n" not in (folder / "3.yaml").read_bytes()


def test_the_command_refuses_an_unknown_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "selection.yaml").write_text(
        f"model: {SPEC}\ncases:\n- {{id: '999', tags: [], note: n}}\n", encoding="utf-8"
    )

    result = runner.invoke(cli.app, ["suite"])

    assert result.exit_code == 2
    assert "999" in result.output


def test_the_command_refuses_a_selection_that_isnt_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "selection.yaml").write_text("cases: [unclosed\n", encoding="utf-8")

    result = runner.invoke(cli.app, ["suite"])

    assert result.exit_code == 2
    assert "isn't valid YAML" in result.output


# ---- the committed suite ---------------------------------------------------------


def test_the_committed_suite_matches_the_dataset() -> None:
    expected = suite.build_suite(
        load_dataset(DEFAULT_DATASET), suite.load_selection(ROOT / "evals" / "selection.yaml")
    )
    folder = ROOT / "evals" / "cases" / "triager"
    found = {p.name: p.read_text(encoding="utf-8") for p in folder.glob("*.yaml")}

    assert sorted(found) == sorted(expected), "run `uv run triager suite`"
    differ = sorted(name for name, text in expected.items() if found[name] != text)
    assert not differ, f"{', '.join(differ)} differ from the dataset: run `uv run triager suite`"
