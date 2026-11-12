import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from issue_triager import export
from issue_triager.dataset import (
    DEFAULT_DATASET,
    GroundTruth,
    IssueExport,
    RepoInfo,
    TriageCase,
    dump_export,
    dump_repo,
    load_dataset,
)

DATA = load_dataset()
REAL = DATA.corpus
SYNTHETIC = DATA.synthetic
WEEK_1 = "Week 1: Demo agent and foundations"
# An issue that's still open in the bundled data. Found, not hard-coded: re-exports close them.
STILL_OPEN = next(
    c.id
    for c in DATA.corpus.values()
    if c.state != "closed" and not c.truth.duplicate_of and c.id != "19"  # 19: the triaged one
)


def gh_issue(number: int, labels: list[str], milestone: str | None, **extra: Any) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Issue {number}",
        "body": "Body\r\nline two\n",
        "labels": [{"name": name} for name in labels],
        "milestone": {"title": milestone} if milestone else None,
        "state": "OPEN",
        "stateReason": "",
        "comments": [],
        **extra,
    }


# ---- The bundled dataset (this project's own issues) -----------------------------


def test_real_export_has_known_duplicate() -> None:
    assert REAL["3"].truth.duplicate_of == "2"
    assert REAL["3"].state == "closed"
    assert REAL["2"].truth.duplicate_of is None


def test_case_ids_are_unique() -> None:
    ids = [c.id for c in DATA.cases]

    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", DATA.cases, ids=lambda c: c.id)
def test_ground_truth_follows_the_repo_taxonomy_and_rules(case: TriageCase) -> None:
    assert DATA.problems(case.truth, target_id=case.id) == []


def test_rules_are_loaded() -> None:
    assert DATA.rules.require_label is True
    assert DATA.rules.labels_needing_milestone == {"stretch"}


@pytest.mark.parametrize("case", [c for c in DATA.cases if c.truth.regression_of], ids=str)
def test_regression_keeps_the_original_milestone(case: TriageCase) -> None:
    original = REAL[str(case.truth.regression_of)]

    assert case.truth.milestone == original.truth.milestone


def test_synthetic_set_separates_duplicates_from_regressions() -> None:
    regressions = [c for c in SYNTHETIC if c.truth.regression_of]
    dupes_of_done_work = [
        c
        for c in SYNTHETIC
        if c.truth.duplicate_of and REAL[c.truth.duplicate_of].state == "closed"
    ]

    assert len(regressions) >= 3
    assert len(dupes_of_done_work) >= 3


def test_about_sixty_synthetic_cases_of_both_kinds() -> None:
    assert 55 <= len(SYNTHETIC) <= 70
    assert {c.kind for c in SYNTHETIC} == {"paraphrase", "tricky"}
    assert all(c.source == "synthetic" and c.id.startswith("s") for c in SYNTHETIC)


@pytest.mark.parametrize("case", [c for c in SYNTHETIC if c.kind == "paraphrase"], ids=str)
def test_paraphrase_matches_its_source_issue(case: TriageCase) -> None:
    source = REAL[str(case.based_on)]

    assert case.truth.duplicate_of == case.based_on
    assert case.truth.model_dump(exclude={"duplicate_of"}) == source.truth.model_dump(
        exclude={"duplicate_of"}
    )
    assert case.title != source.title


def test_tricky_cases_explain_themselves() -> None:
    tricky = [c for c in SYNTHETIC if c.kind == "tricky"]

    assert all(c.note for c in tricky)
    assert all(c.based_on is None for c in tricky)


def test_agent_input_hides_the_answer() -> None:
    assert set(REAL["6"].agent_input().model_dump()) == {"id", "title", "body"}


def test_committed_files_are_in_canonical_form() -> None:
    """Re-exporting an unchanged repo must produce no diff."""
    issues = (DEFAULT_DATASET / "issues.json").read_text(encoding="utf-8")
    repo = (DEFAULT_DATASET / "repo.json").read_text(encoding="utf-8")

    assert dump_export(IssueExport.model_validate_json(issues)) == issues
    assert dump_repo(RepoInfo.model_validate_json(repo)) == repo


# ---- Rules -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("truth", "expected"),
    [
        ({"labels": ["ui"], "milestone": WEEK_1}, None),
        ({"labels": [], "milestone": WEEK_1}, "at least one label"),
        ({"labels": ["wontfix"], "milestone": None}, "Unknown labels"),
        ({"labels": ["ui"], "milestone": "Week 9"}, "Unknown milestone"),
        ({"labels": ["ui", "stretch"], "milestone": None}, "need a milestone"),
        ({"labels": ["ui"], "milestone": None, "duplicate_of": "29"}, "being triaged"),
        ({"labels": ["ui"], "milestone": None, "duplicate_of": "999"}, "doesn't exist"),
        ({"labels": ["ui"], "milestone": None, "regression_of": STILL_OPEN}, "isn't done"),  # open
        ({"labels": ["ui"], "milestone": None, "regression_of": "3"}, "isn't done"),  # a dupe
        ({"labels": ["ui"], "milestone": None, "regression_of": "1"}, None),
    ],
)
def test_rules(truth: dict[str, Any], expected: str | None) -> None:
    found = DATA.problems(GroundTruth.model_validate(truth), target_id="29")

    if expected is None:
        assert found == []
    else:
        assert any(expected in p for p in found), found


def test_duplicate_and_regression_are_exclusive() -> None:
    with pytest.raises(ValueError, match="not both"):
        GroundTruth(labels=frozenset({"ui"}), milestone=None, duplicate_of="2", regression_of="1")


def test_a_dataset_without_extras_loads(tmp_path: Path) -> None:
    (tmp_path / "repo.json").write_text(json.dumps({"repo": "a/b"}), encoding="utf-8")
    (tmp_path / "issues.json").write_text(json.dumps({"repo": "a/b"}), encoding="utf-8")

    data = load_dataset(tmp_path)

    assert data.synthetic == []
    assert data.rules.require_label is False
    assert data.problems(GroundTruth(labels=frozenset(), milestone=None), target_id="1") == []


# ---- Exporter --------------------------------------------------------------------


def test_export_maps_labels_and_milestone_to_ground_truth() -> None:
    issues = [gh_issue(6, ["demo-agent", "stretch"], WEEK_1)]

    (case,) = export.build_export(issues, "a/b").issues

    assert case.truth.labels == {"demo-agent", "stretch"}
    assert case.truth.milestone == WEEK_1
    assert case.body == "Body\nline two"


def test_export_keeps_any_label() -> None:
    (case,) = export.build_export([gh_issue(1, ["good first issue", "bug"], None)], "a/b").issues

    assert case.truth.labels == {"good first issue", "bug"}


def test_export_reads_duplicate_from_comment() -> None:
    dupe = gh_issue(3, ["ui"], None, state="CLOSED", comments=[{"body": "Duplicate of #2 (oops)."}])

    (case,) = export.build_export([dupe], "a/b").issues

    assert case.truth.duplicate_of == "2"
    assert case.state == "closed"


def test_export_reads_regression_from_body() -> None:
    bug = gh_issue(40, ["infra"], None, body="CI broke again.\n\nRegression of #1\n")

    (case,) = export.build_export([bug], "a/b").issues

    assert case.truth.regression_of == "1"


def test_export_handles_empty_body() -> None:
    (case,) = export.build_export([gh_issue(1, [], None, body=None)], "a/b").issues

    assert case.body == ""


def test_export_sorts_by_issue_number() -> None:
    issues = [gh_issue(n, ["ui"], None) for n in (10, 2, 33)]

    assert [c.id for c in export.build_export(issues, "a/b").issues] == ["2", "10", "33"]


def fake_gh(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []
    replies = {
        ("repo", "view"): {"description": "A test repo"},
        ("label", "list"): [{"name": "ui", "description": "UI"}, {"name": "bug"}],
        ("api",): [
            {"title": "Later", "number": 2, "due_on": "2027-02-01T00:00:00Z", "state": "open"},
            {"title": "Sooner", "number": 1, "due_on": "2027-01-01T00:00:00Z", "state": "open"},
        ],
        ("issue", "list"): [gh_issue(1, ["ui"], "Sooner")],
    }

    def fake_run(args: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        key = next(k for k in replies if tuple(args[1 : 1 + len(k)]) == k)
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(replies[key]))

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_export_writes_a_loadable_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_gh(monkeypatch)

    export.export("a/b", tmp_path)
    data = load_dataset(tmp_path)

    assert data.repo.description == "A test repo"
    assert [m.title for m in data.repo.milestones] == ["Sooner", "Later"]  # by due date
    assert data.real[0].truth.milestone == "Sooner"


def test_export_only_reads_from_github(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = fake_gh(monkeypatch)

    export.export("a/b", tmp_path)

    read_only = {("gh", "repo", "view"), ("gh", "label", "list"), ("gh", "issue", "list")}
    for args in calls:
        assert tuple(args[:3]) in read_only or (args[1] == "api" and "-X" not in args), args


def test_cases_about_unfinished_work_point_at_open_issues() -> None:
    """A re-export can close an issue a case calls unfinished, leaving its truth stale."""
    import re

    from issue_triager.tools import status_of

    for case in SYNTHETIC:
        for number in re.findall(r"#(\d+) isn't finished", case.note):
            assert status_of(REAL[number]) == "open", f"{case.id}: #{number} is done now"
