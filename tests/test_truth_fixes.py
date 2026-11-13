"""Truth fixes from AFR #14's hand error analysis: overrides for real issues' truths,
accepted alternatives for debatable ones, and s65 no longer a regression.

Written from .review/14-truth-fixes/spec.md before the code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from issue_triager.dataset import GroundTruth, TriageCase, load_dataset

WEEK_2 = "Week 2: Recorder core"


def write_dataset(tmp: Path, *, overrides: Any = None, synthetic: list[Any] | None = None) -> Path:
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
    if overrides is not None:
        (tmp / "truth_overrides.yaml").write_text(yaml.safe_dump(overrides), encoding="utf-8")
    return tmp


PARAPHRASE = {"id": "s07", "kind": "paraphrase", "based_on": "7", "title": "Schema please",
              "body": "b", "truth": {"labels": ["recorder"], "milestone": WEEK_2,
                                     "duplicate_of": "7"}}  # fmt: skip


# --- overrides -----------------------------------------------------------------------


def test_an_override_replaces_the_fields_it_gives_and_records_why(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, overrides={"overrides": {
        "7": {"labels": ["infra", "recorder"], "reason": "AFR #14: migrations are infra too"},
    }})  # fmt: skip

    data = load_dataset(path)

    fixed = data.by_id["7"]
    assert fixed.truth.labels == frozenset({"infra", "recorder"})
    assert fixed.truth.milestone == WEEK_2  # not given: kept
    assert fixed.truth_override == "AFR #14: migrations are infra too"
    assert data.by_id["8"].truth_override == ""


def test_a_paraphrase_follows_its_sources_override(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, synthetic=[PARAPHRASE], overrides={"overrides": {
        "7": {"labels": ["infra", "recorder"], "reason": "r"},
    }})  # fmt: skip

    paraphrase = load_dataset(path).by_id["s07"]

    assert paraphrase.truth.labels == frozenset({"infra", "recorder"})
    assert paraphrase.truth.duplicate_of == "7"  # its own link stays
    assert paraphrase.truth_override == "r"


@pytest.mark.parametrize(
    "overrides",
    [
        {"overrides": {"7": {"labels": ["infra"]}}},  # no reason
        {"overrides": {"99": {"labels": ["infra"], "reason": "r"}}},  # no such issue
        {"overrides": {"s07": {"labels": ["infra"], "reason": "r"}}},  # synthetic: edit it
        {"overrides": {"7": {"colour": "red", "reason": "r"}}},  # not a truth field
    ],
)
def test_a_bad_override_is_refused(tmp_path: Path, overrides: Any) -> None:
    path = write_dataset(tmp_path, synthetic=[PARAPHRASE], overrides=overrides)

    with pytest.raises(ValueError):
        load_dataset(path)


def test_the_bundled_overrides_are_the_ones_decided_in_afr_14() -> None:
    data = load_dataset()
    labels = {i: sorted(data.by_id[i].truth.labels) for i in ("1", "7", "17", "20", "26", "33")}

    assert labels == {
        "1": ["documentation", "infra"],
        "7": ["infra", "recorder"],
        "17": ["documentation", "evals"],
        "20": ["documentation", "evals"],
        "26": ["infra", "mcp"],
        "33": ["documentation", "infra"],
    }
    assert all(data.by_id[i].truth_override for i in labels)
    assert sorted(data.by_id["s19"].truth.labels) == ["documentation", "evals"]  # paraphrase of 20


def test_the_override_reason_goes_on_the_run() -> None:
    from issue_triager.grading import grade_attributes

    case = load_dataset().by_id["26"]
    decision = {"labels": ["infra", "mcp"], "milestone": case.truth.milestone,
                "duplicate_of": None, "regression_of": None}  # fmt: skip

    marks = grade_attributes({"status": "applied", "decision": decision}, case)

    assert marks["triager.grade"] == "correct"
    assert marks["triager.truth_override"] == case.truth_override


def test_a_run_without_an_override_has_no_override_attribute() -> None:
    from issue_triager.grading import grade_attributes

    case = load_dataset().by_id["8"]

    assert "triager.truth_override" not in grade_attributes({"status": "failed"}, case)


# --- accepted alternatives -------------------------------------------------------------


def test_an_accepted_alternative_counts_as_right() -> None:
    from issue_triager.grading import grade_fields

    accept = {"duplicate_of": [None], "labels": [["evals", "analyst"]]}
    truth = GroundTruth(labels=frozenset({"evals"}), milestone=None, duplicate_of="23",
                        accept=accept)  # fmt: skip
    base = {"labels": ["evals"], "milestone": None, "regression_of": None}
    both = base | {"labels": ["analyst", "evals"], "duplicate_of": None}

    assert grade_fields(base | {"duplicate_of": "#23"}, truth)["duplicate_of"]
    assert grade_fields(base | {"duplicate_of": None}, truth)["duplicate_of"]
    assert not grade_fields(base | {"duplicate_of": "24"}, truth)["duplicate_of"]
    assert grade_fields(both, truth)["labels"]


def test_accept_only_takes_truth_fields() -> None:
    with pytest.raises(ValueError):
        GroundTruth(labels=frozenset(), milestone=None, accept={"colour": ["red"]})


def test_accepted_alternatives_go_on_the_run() -> None:
    from issue_triager.grading import grade_attributes

    case = load_dataset().by_id["s58"]
    decision = {"labels": sorted(case.truth.labels), "milestone": case.truth.milestone,
                "duplicate_of": None, "regression_of": None}  # fmt: skip

    marks = grade_attributes({"status": "applied", "decision": decision}, case)

    assert marks["triager.grade"] == "correct"
    assert marks["triager.expected.duplicate_of"] == "23"
    assert marks["triager.accepted.duplicate_of"] == ["(none)"]


def test_a_case_without_alternatives_has_no_accepted_attributes() -> None:
    from issue_triager.grading import grade_attributes

    marks = grade_attributes({"status": "failed"}, load_dataset().by_id["8"])

    assert not any(key.startswith("triager.accepted.") for key in marks)


# --- s65 -------------------------------------------------------------------------------


def test_s65_is_not_a_regression() -> None:
    case: TriageCase = load_dataset().by_id["s65"]

    assert case.truth.regression_of is None
    assert sorted(case.truth.labels) == ["demo-agent"]
    assert case.truth.milestone == "Week 1: Demo agent and foundations"
    assert "regression of #2" not in case.note.lower()
