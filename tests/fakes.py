"""Fake chat models for tests: scripted replies, no provider, no key."""

import uuid
from collections.abc import Iterator
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from issue_triager.graph import Models


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
