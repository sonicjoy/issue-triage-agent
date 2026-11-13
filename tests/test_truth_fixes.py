"""Truth fixes from AFR #14's hand error analysis: accepted answers for under-labelled real
issues (and the synthetic duplicates of them), a debatable duplicate, and s65 no longer a
regression. The exported truth, and what the agent's tools show, never change.

Written from .review/14-truth-fixes/spec.md (round 1: accept both) before the code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from issue_triager.dataset import Accepted, GroundTruth, TriageCase, load_dataset

WEEK_2 = "Week 2: Recorder core"


def write_dataset(tmp: Path, *, accepted: Any = None, synthetic: list[Any] | None = None) -> Path:
    repo = {"repo": "a/b", "labels": [{"name": n} for n in ("infra", "mcp", "recorder")],
            "milestones": [{"title": WEEK_2}]}  # fmt: skip
    issues = {"repo": "a/b", "issues": [
        {"id": "7", "title": "Trace schema", "body": "b", "source": "real", "state": "closed",
         "truth": {"labels": ["recorder"], "milestone": WEEK_2}},
        {"id": "8", "title": "Other", "body": "b", "source": "real",
         "truth": {"labels": ["infra"], "milestone": None}},
    ]}  # fmt: skip
    (tmp / "repo.json").write_text(json.dumps(repo), encoding="utf-8")
    (tmp / "issues.json").write_text(json.dumps(issues), encoding="utf-8")
    if synthetic is not None:
        (tmp / "synthetic.yaml").write_text(yaml.safe_dump({"cases": synthetic}), encoding="utf-8")
    if accepted is not None:
        (tmp / "accepted_answers.yaml").write_text(yaml.safe_dump(accepted), encoding="utf-8")
    return tmp


PARAPHRASE = {"id": "s07", "kind": "paraphrase", "based_on": "7", "title": "Schema please",
              "body": "b", "truth": {"labels": ["recorder"], "milestone": WEEK_2,
                                     "duplicate_of": "7"}}  # fmt: skip
TRICKY_DUPLICATE = {"id": "s70", "kind": "tricky", "note": "same work, other words",
                    "title": "Migrations", "body": "b",
                    "truth": {"labels": ["recorder"], "milestone": WEEK_2,
                              "duplicate_of": "7"}}  # fmt: skip
FIX_7 = {"accept": {"7": {"labels": [["infra", "recorder"]], "reason": "migrations are infra"}}}


# --- accepted answers for real issues ----------------------------------------------------


def test_an_accepted_answer_adds_to_the_truth_and_replaces_nothing(tmp_path: Path) -> None:
    data = load_dataset(write_dataset(tmp_path, accepted=FIX_7))

    fixed = data.by_id["7"]
    assert fixed.truth.labels == frozenset({"recorder"})  # as exported: tools show this
    assert fixed.truth.accept.labels == (frozenset({"infra", "recorder"}),)
    assert fixed.accept_reason == "migrations are infra"
    assert data.by_id["8"].accept_reason == ""


def test_every_synthetic_duplicate_of_the_issue_follows(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, synthetic=[PARAPHRASE, TRICKY_DUPLICATE], accepted=FIX_7)

    data = load_dataset(path)

    for case_id in ("s07", "s70"):
        case = data.by_id[case_id]
        assert case.truth.labels == frozenset({"recorder"}), case_id
        assert case.truth.duplicate_of == "7"
        assert case.truth.accept.labels == (frozenset({"infra", "recorder"}),)
        assert case.accept_reason == "migrations are infra"


@pytest.mark.parametrize(
    "accepted",
    [
        {"accept": {"7": {"labels": [["infra"]]}}},  # no reason
        {"accept": {"99": {"labels": [["infra"]], "reason": "r"}}},  # no such issue
        {"accept": {"s07": {"labels": [["infra"]], "reason": "r"}}},  # synthetic: edit it
        {"accept": {"7": {"colour": ["red"], "reason": "r"}}},  # not a truth field
        {"accept": {"7": {"labels": ["infra"], "reason": "r"}}},  # a flat list, not sets
        {"accept": {"7": {"duplicate_of": [24], "reason": "r"}}},  # an unquoted id
        {"overrides": {"7": {"labels": [["infra"]], "reason": "r"}}},  # misspelt top level
    ],
)
def test_a_bad_entry_is_refused(tmp_path: Path, accepted: Any) -> None:
    path = write_dataset(tmp_path, synthetic=[PARAPHRASE], accepted=accepted)

    with pytest.raises(ValueError):
        load_dataset(path)


def test_the_bundled_answers_are_the_ones_decided_in_afr_14() -> None:
    data = load_dataset()
    accepted = {i: [sorted(s) for s in data.by_id[i].truth.accept.labels]
                for i in ("1", "7", "17", "20", "26", "33")}  # fmt: skip

    assert accepted == {
        "1": [["documentation", "infra"]],
        "7": [["infra", "recorder"]],
        "17": [["documentation", "evals"]],
        "20": [["documentation", "evals"]],
        "26": [["infra", "mcp"]],
        "33": [["documentation", "infra"]],
    }
    assert all(data.by_id[i].accept_reason for i in accepted)
    assert data.by_id["s19"].truth.accept.labels == data.by_id["20"].truth.accept.labels
    assert data.by_id["s57"].truth.accept.labels == data.by_id["33"].truth.accept.labels


def test_accepted_labels_follow_the_repo_taxonomy() -> None:
    data = load_dataset()
    known = data.repo.label_names

    for case in data.cases:
        for labels in case.truth.accept.labels:
            assert labels <= known, (case.id, sorted(labels - known))


def test_the_agents_tools_still_show_the_exported_labels() -> None:
    from issue_triager.dataset import DEFAULT_DATASET, IssueExport

    raw = (DEFAULT_DATASET / "issues.json").read_text("utf-8")
    exported = IssueExport.model_validate_json(raw)
    corpus = load_dataset().corpus

    for issue in exported.issues:
        assert corpus[issue.id].truth.labels == issue.truth.labels


# --- grading -------------------------------------------------------------------------------


def test_an_accepted_answer_counts_as_right() -> None:
    from issue_triager.grading import grade_fields

    accept = Accepted(duplicate_of=(None,), labels=(frozenset({"evals", "analyst"}),))
    truth = GroundTruth(labels=frozenset({"evals"}), milestone=None, duplicate_of="23",
                        accept=accept)  # fmt: skip
    base = {"labels": ["evals"], "milestone": None, "regression_of": None}
    both = base | {"labels": ["analyst", "evals"], "duplicate_of": None}

    assert grade_fields(base | {"duplicate_of": "#23"}, truth)["duplicate_of"]
    assert grade_fields(base | {"duplicate_of": None}, truth)["duplicate_of"]
    assert not grade_fields(base | {"duplicate_of": "24"}, truth)["duplicate_of"]
    assert grade_fields(both, truth)["labels"]
    assert grade_fields(base | {"duplicate_of": None}, truth)["labels"]  # the truth itself


def test_a_run_matching_the_exported_truth_stays_right() -> None:
    from issue_triager.grading import grade_attributes

    case = load_dataset().by_id["26"]
    exported = {"labels": sorted(case.truth.labels), "milestone": case.truth.milestone,
                "duplicate_of": None, "regression_of": None}  # fmt: skip

    marks = grade_attributes({"status": "applied", "decision": exported}, case)

    assert marks["triager.grade"] == "correct"


def test_accepted_answers_and_why_go_on_the_run() -> None:
    from issue_triager.grading import grade_attributes

    case = load_dataset().by_id["26"]
    fuller = {"labels": ["infra", "mcp"], "milestone": case.truth.milestone,
              "duplicate_of": None, "regression_of": None}  # fmt: skip

    marks = grade_attributes({"status": "applied", "decision": fuller}, case)

    assert marks["triager.grade"] == "correct"
    assert marks["triager.expected.labels"] == ["infra"]
    assert marks["triager.accepted.labels"] == ["infra, mcp"]
    assert marks["triager.accept_reason"] == case.accept_reason


def test_s58_accepts_no_duplicate() -> None:
    from issue_triager.grading import grade_attributes

    case = load_dataset().by_id["s58"]
    decision = {"labels": sorted(case.truth.labels), "milestone": case.truth.milestone,
                "duplicate_of": None, "regression_of": None}  # fmt: skip

    marks = grade_attributes({"status": "applied", "decision": decision}, case)

    assert marks["triager.grade"] == "correct"
    assert marks["triager.expected.duplicate_of"] == "23"
    assert marks["triager.accepted.duplicate_of"] == ["(none)"]


def test_a_case_without_accepted_answers_has_no_such_attributes() -> None:
    from issue_triager.grading import grade_attributes

    marks = grade_attributes({"status": "failed"}, load_dataset().by_id["8"])

    assert not any(k.startswith("triager.accept") for k in marks)


# --- s65 -------------------------------------------------------------------------------


def test_s65_is_not_a_regression() -> None:
    case: TriageCase = load_dataset().by_id["s65"]

    assert case.truth.regression_of is None
    assert sorted(case.truth.labels) == ["demo-agent"]
    assert case.truth.milestone == "Week 1: Demo agent and foundations"
    assert "regression of #2" not in case.note.lower()
