"""Grade a triage against the issue's real triage, which the agent never sees.

The grade goes on the run's root span (`grade_attributes`), so a recorder such as Agent
Flight Recorder stores it with the run and wrong runs can be found later. It's set only
after the triage has finished, never on the agent's own spans or in its prompt.
"""

from __future__ import annotations

from typing import Any

from issue_triager.dataset import GroundTruth, TriageCase

FIELDS = ("labels", "milestone", "duplicate_of", "regression_of")


def grade_fields(decision: dict[str, Any], truth: GroundTruth) -> dict[str, bool]:
    """Whether each field of the decision matches the ground truth."""
    marks: dict[str, bool] = {}
    for name in FIELDS:
        got, want = decision.get(name), getattr(truth, name)
        if name == "labels":
            got, want = sorted(got or []), sorted(want)
        elif isinstance(got, str):
            got = got.lstrip("#")
        marks[name] = got == want
    return marks


def grade_attributes(state: dict[str, Any], case: TriageCase) -> dict[str, str]:
    """Span attributes for a finished triage: `triager.grade` is `correct`, `wrong` (some
    field differs from the truth) or `failed` (no valid decision), plus one ok/wrong per
    field when there was a decision."""
    attributes = {"triager.case_source": case.source}
    decision = state.get("decision")
    if state.get("status") == "failed" or not isinstance(decision, dict):
        return attributes | {"triager.grade": "failed"}
    marks = grade_fields(decision, case.truth)  # pyright: ignore[reportUnknownArgumentType]
    attributes |= {f"triager.grade.{name}": "ok" if ok else "wrong" for name, ok in marks.items()}
    return attributes | {"triager.grade": "correct" if all(marks.values()) else "wrong"}
