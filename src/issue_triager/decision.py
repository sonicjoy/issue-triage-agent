"""The triager's structured output.

The schema is built per repo: labels and milestones become enums of that repo's
taxonomy, so a provider's structured output can only pick values that exist.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from issue_triager.dataset import GroundTruth, RepoInfo


class TriageDecision(BaseModel):
    """Your triage decision for the issue."""

    labels: list[str] = Field(description="Labels to apply.")
    milestone: str | None = Field(description="Milestone the work belongs in, or null.")
    duplicate_of: str | None = Field(
        default=None,
        description="Id of an existing issue asking for the same work, if you are fairly sure.",
    )
    regression_of: str | None = Field(
        default=None,
        description="Id of a done (closed) issue whose work this reports as broken again.",
    )
    confidence: float = Field(ge=0, le=1, description="How sure you are overall, 0 to 1.")
    rationale: str = Field(description="One or two sentences explaining the decision.")

    def to_truth(self) -> GroundTruth:
        """The comparable part, in the same shape as a dataset's ground truth."""
        return GroundTruth(
            labels=frozenset(self.labels),
            milestone=self.milestone,
            duplicate_of=self.duplicate_of.lstrip("#") if self.duplicate_of else None,
            regression_of=self.regression_of.lstrip("#") if self.regression_of else None,
        )


def _enum(values: list[str]) -> Any:
    return Literal[tuple(values)]  # pyright: ignore[reportInvalidTypeForm]  # built at runtime


def decision_model(repo: RepoInfo) -> type[TriageDecision]:
    """`TriageDecision` narrowed to `repo`'s labels and milestones."""
    fields: dict[str, Any] = {}
    if repo.labels:
        names = [label.name for label in repo.labels]
        fields["labels"] = (list[_enum(names)], Field(description="Labels to apply."))
    if repo.milestones:
        titles = [m.title for m in repo.milestones]
        fields["milestone"] = (
            _enum(titles) | None,
            Field(description="Milestone the work belongs in, or null if none fits."),
        )
    else:
        fields["milestone"] = (None, Field(default=None, description="This repo has none."))
    return create_model(
        "TriageDecision", __base__=TriageDecision, __doc__=TriageDecision.__doc__, **fields
    )
