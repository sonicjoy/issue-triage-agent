"""Grade a triage against the issue's real triage, which the agent never sees.

The grade goes on the run's root span (`grade_attributes`), so a recorder such as Agent
Flight Recorder stores it with the run and wrong runs can be found later. It's set only
after the triage has finished, never on the agent's own spans or in its prompt. So are what
the agent decided and what the truth is (`triager.decided.*`, `triager.expected.*`), for
someone reading the run.
"""

from __future__ import annotations

from typing import Any

from issue_triager.dataset import GroundTruth, TriageCase

FIELDS = ("labels", "milestone", "duplicate_of", "regression_of")
NONE = "(none)"  # span attributes can't be null

Attribute = str | list[str]


def _normal(name: str, value: Any) -> Any:
    """A field's value as it's compared: labels sorted, issue numbers without the #."""
    if name == "labels":
        return sorted(value or [])
    if isinstance(value, str):
        return value.lstrip("#")
    return value


def grade_fields(decision: dict[str, Any], truth: GroundTruth) -> dict[str, bool]:
    """Whether each field of the decision matches the ground truth."""
    return {
        name: _normal(name, decision.get(name)) == _normal(name, getattr(truth, name))
        for name in FIELDS
    }


def _shown(prefix: str, values: dict[str, Any]) -> dict[str, Attribute]:
    shown: dict[str, Attribute] = {}
    for name in FIELDS:
        value = _normal(name, values.get(name))
        shown[f"{prefix}.{name}"] = value if name == "labels" else (value or NONE)
    return shown


def grade_attributes(state: dict[str, Any], case: TriageCase) -> dict[str, Attribute]:
    """Span attributes for a finished triage: `triager.grade` is `correct`, `wrong` (some
    field differs from the truth) or `failed` (no valid decision), plus one ok/wrong per
    field when there was a decision. `triager.expected.*` holds the truth, and
    `triager.decided.*` the decision when there was one."""
    truth = {name: getattr(case.truth, name) for name in FIELDS}
    attributes: dict[str, Attribute] = {"triager.case_source": case.source}
    attributes |= _shown("triager.expected", truth)
    decision = state.get("decision")
    if state.get("status") == "failed" or not isinstance(decision, dict):
        return attributes | {"triager.grade": "failed"}
    decided: dict[str, Any] = decision  # pyright: ignore[reportUnknownVariableType]
    marks = grade_fields(decided, case.truth)
    attributes |= _shown("triager.decided", decided)
    attributes |= {f"triager.grade.{name}": "ok" if ok else "wrong" for name, ok in marks.items()}
    return attributes | {"triager.grade": "correct" if all(marks.values()) else "wrong"}
