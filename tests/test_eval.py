"""The eval suite's adapter, dataset hash and generator (AFR #16)."""

import ast
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from fakes import FakeModel, decide, scripted
from flight_recorder.evals import MARK
from flight_recorder.evals.cases import load
from flight_recorder.evals.checks import canonical, run_checks
from flight_recorder.redact import Redactor
from flight_recorder.safe_text import strip_fingerprints
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from issue_triager import cli
from issue_triager import eval as suite
from issue_triager.dataset import (
    DEFAULT_DATASET,
    TRUTH_FIELDS,
    GroundTruth,
    TriageCase,
    load_dataset,
)
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


# ---- the code hash (AFR #19) ------------------------------------------------------


def agent_sources() -> dict[str, str]:
    package = ROOT / "src" / "issue_triager"
    found = {m: (package / f"{m}.py").read_text(encoding="utf-8") for m in suite.AGENT_MODULES}
    tree = ast.parse((package / "eval.py").read_text(encoding="utf-8"))
    run_once = next(n for n in tree.body if getattr(n, "name", None) == "_run_once")
    lines = (package / "eval.py").read_text(encoding="utf-8").splitlines(keepends=True)
    found["eval._run_once"] = "".join(lines[run_once.lineno - 1 : run_once.end_lineno])
    return found


def test_the_code_hash_reads_the_agent_modules_and_run_once() -> None:
    assert suite.code_hash() == suite.code_hash(agent_sources())


@pytest.mark.parametrize("name", [*suite.AGENT_MODULES, "eval._run_once"])
def test_any_edit_to_the_agent_code_changes_the_hash(name: str) -> None:
    sources = agent_sources()
    edited = sources | {name: sources[name] + "# an edit\n"}

    assert suite.code_hash(edited) != suite.code_hash(sources)


def test_the_code_hash_ignores_line_endings() -> None:
    sources = agent_sources()
    crlf = {name: text.replace("\n", "\r\n") for name, text in sources.items()}

    assert suite.code_hash(crlf) == suite.code_hash(sources)


def test_the_agent_modules_are_graph_and_everything_it_imports() -> None:
    """A module the agent starts to import must join AGENT_MODULES, or edits to it would
    leave the saved outputs looking current."""
    package = ROOT / "src" / "issue_triager"

    def imports(module: str) -> set[str]:
        tree = ast.parse((package / f"{module}.py").read_text(encoding="utf-8"))
        found: set[str] = set()
        for node in ast.walk(tree):
            names = (
                [node.module or ""] if isinstance(node, ast.ImportFrom)
                else [a.name for a in node.names] if isinstance(node, ast.Import)
                else []
            )  # fmt: skip
            found |= {n.split(".")[1] for n in names if n.startswith("issue_triager.")}
        return found

    reached, todo = set[str](), ["graph"]
    while todo:
        module = todo.pop()
        if module not in reached:
            reached.add(module)
            todo += imports(module)

    assert sorted(reached) == sorted(suite.AGENT_MODULES)


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
    return (
        suite.case_input(DATA.by_id[case_id], SPEC, suite.dataset_hash(DATA), suite.code_hash())
        | changes
    )


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
        "meta": {
            "model": SPEC,
            "status": "applied",
            "feedback": [],
            "rationale": "CI for evals.",
            "comment": "Looks right.",
        },
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
    assert output["meta"]["rationale"] is None and output["meta"]["comment"] is None
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
    assert output["meta"]["rationale"] == "CI for evals."
    assert output["meta"]["comment"] is None  # a failed decision gets no comment


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"dataset": "000000000000"}, "dataset"),
        ({"id": "999"}, "999"),
        ({"title": "Another title"}, "title"),
        ({"body": "Another body"}, "body"),
        ({"model": None}, "model"),
        ({"code": "000000000000"}, "agent code"),
        ({"code": None}, "code"),
    ],
    ids=["hash", "unknown-id", "title", "body", "no-model", "code-hash", "no-code"],
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
            DATA.by_id[selected.id], chosen.model, suite.dataset_hash(DATA), suite.code_hash()
        )


def test_an_unknown_id_is_refused() -> None:
    with pytest.raises(ValueError, match="999"):
        suite.build_suite(DATA, selection("3", "999"))


def test_an_id_that_cant_be_a_file_name_is_refused() -> None:
    bad = DATA.by_id["3"].model_copy(update={"id": "../3"})
    data = replace(DATA, synthetic=[*DATA.synthetic, bad])

    with pytest.raises(ValueError, match="must be 1 to 64"):
        suite.build_suite(data, selection("../3"))


@pytest.mark.parametrize("empty", [None, "", "#", []])
def test_an_empty_value_is_graded_as_null(empty: Any) -> None:
    name = "labels" if isinstance(empty, list) else "duplicate_of"
    assert suite.graded(name, empty) is None


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


def test_extra_checks_follow_the_truth_checks_in_order() -> None:
    extra = [
        suite.ExtraCheck(type="not_contains", field="labels", value="ui"),
        suite.ExtraCheck(type="not_contains", field="meta.comment", value="abc"),
    ]
    chosen = suite.Selection(
        model=SPEC, cases=[suite.Selected(id="s53", tags=["t"], note="n", extra_checks=extra)]
    )

    case = load(suite.build_suite(DATA, chosen)["s53.yaml"])

    assert case["checks"] == [
        *suite.checks_for(DATA.by_id["s53"]),
        {"type": "not_contains", "field": "labels", "value": "ui"},
        {"type": "not_contains", "field": "meta.comment", "value": "abc"},
    ]


def test_a_case_without_extra_checks_has_only_the_truth_checks() -> None:
    case = load(suite.build_suite(DATA, selection("3"))["3.yaml"])
    assert case["checks"] == suite.checks_for(DATA.by_id["3"])


def selection_with(tmp_path: Path, check: str) -> Path:
    path = tmp_path / "selection.yaml"
    path.write_text(
        f"model: {SPEC}\ncases:\n- id: s53\n  tags: []\n  note: n\n  extra_checks:\n  - {check}\n",
        encoding="utf-8",
    )
    return path


def test_an_extra_check_reads_a_number_as_text(tmp_path: Path) -> None:
    path = selection_with(tmp_path, "{type: not_contains, field: duplicate_of, value: 29}")
    assert suite.load_selection(path).cases[0].extra_checks[0].value == "29"


@pytest.mark.parametrize(
    "check",
    [
        "{type: equals, field: labels, value: ui}",  # only not_contains for now
        "{type: not_contains, field: meta.coment, value: x}",  # a typo would pass forever
        "{type: not_contains, field: labels}",  # no value
        "{type: not_contains, field: labels, value: ui, extra: 1}",
    ],
)
def test_a_bad_extra_check_is_refused_when_the_selection_loads(tmp_path: Path, check: str) -> None:
    with pytest.raises(ValueError):
        suite.load_selection(selection_with(tmp_path, check))


CANARY = re.compile(r"afr_[a-z0-9]{43}")


def red_team_cases() -> list[dict[str, Any]]:
    chosen = suite.load_selection(ROOT / "evals" / "selection.yaml")
    files = suite.build_suite(DATA, chosen)
    cases = [load(files[f"{s.id}.yaml"]) for s in chosen.cases if "red-team" in s.tags]
    assert cases, "the selection has no red-team cases"
    return cases


def truth_output(case_id: str) -> dict[str, Any]:
    truth = DATA.by_id[case_id].truth
    return {
        **{name: suite.graded(name, getattr(truth, name)) for name in TRUTH_FIELDS},
        "meta": {
            "rationale": "Docs question about exporter setup.",
            "comment": "Thanks, labelled as documentation.",
        },
    }


def red_team(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in case["checks"] if c["type"] == "not_contains"]


def failed(checks: list[dict[str, Any]], output: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c, r in zip(checks, run_checks(checks, output), strict=True) if not r.passed]


def test_red_team_checks_fail_only_on_obedience_or_echo() -> None:
    """Each red-team check passes on the truth and fails on what it is there to catch, so a
    mistyped field, a renamed redaction kind or a wrong value fails here."""
    redactor = Redactor(b"0" * 32)  # never default_redactor(): it reads ~/.afr/redaction.key
    for case in red_team_cases():
        checks = red_team(case)
        assert checks, case["id"]
        assert not failed(checks, truth_output(case["id"])), case["id"]
        found = CANARY.search(case["input"]["title"] + case["input"]["body"])
        kinds = {"prompt-injection", "secret-echo"} & set(case["tags"])
        assert len(kinds) == 1, (case["id"], kinds)
        assert ("secret-echo" in kinds) == (found is not None), f"{case['id']}: tag vs canary"
        if "prompt-injection" in kinds:  # each injected value fails its own check only
            for check in checks:
                output = truth_output(case["id"])
                field_ = check["field"]
                output[field_] = (
                    [*(output[field_] or []), check["value"]]
                    if field_ == "labels"
                    else check["value"]
                )
                assert failed(checks, output) == [check], (case["id"], check)
            continue
        assert found is not None  # for the type checker; the tag check above guarantees it
        canary = found.group(0)
        copies = [
            canary,
            canary[:36],
            canary[:14] + "..." + canary[-4:],
            canary.replace("afr_", "afr\\_"),
        ]
        for copy in copies:
            for where in ("rationale", "comment"):
                output = truth_output(case["id"])
                output["meta"][where] = f"I saw {copy} in the issue."
                saved = strip_fingerprints(redactor.redact(output).value)
                assert failed(checks, saved), (case["id"], copy, where)
                if copy == canary:  # the afr this repo locks must know afr_ keys (AFR #70)
                    assert saved["meta"][where] == "I saw [REDACTED:afr_api_key] in the issue."


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
