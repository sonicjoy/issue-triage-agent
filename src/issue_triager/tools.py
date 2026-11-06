"""The triager's tools. They read the dataset only; nothing touches GitHub.

`IssueTools` holds the logic once; `make_tools` (LangChain) and `agents_sdk.py` (OpenAI
Agents SDK) wrap it, so every version of the agent gets the same tools and descriptions.
`FailureInjector` breaks tool calls on purpose, so the recorder sees errors the
agent has to cope with.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from langchain_core.tools import BaseTool, ToolException, tool

from issue_triager.dataset import TriageCase
from issue_triager.search import KeywordIndex

FailureMode = Literal["timeout", "bad-data", "empty", "noise"]
FAILURE_MODES: tuple[FailureMode, ...] = get_args(FailureMode)
BODY_PREVIEW = 300


@dataclass
class FailureInjector:
    """Picks a failure for a tool call with probability `rate`, or None."""

    modes: Sequence[FailureMode] = ()
    rate: float = 0.0
    seed: int | None = None
    injected: list[str] = field(default_factory=list[str])

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def pick(self, tool_name: str) -> FailureMode | None:
        if not self.modes or self._rng.random() >= self.rate:
            return None
        mode = self._rng.choice(self.modes)
        self.injected.append(f"{tool_name}:{mode}")
        return mode

    def shuffle(self, items: list[TriageCase]) -> list[TriageCase]:
        return self._rng.sample(items, k=len(items))


def status_of(case: TriageCase) -> str:
    if case.truth.duplicate_of:
        return f"closed as duplicate of #{case.truth.duplicate_of}"
    return "done (closed)" if case.state == "closed" else "open"


def summarise(case: TriageCase, *, full: bool = False) -> dict[str, Any]:
    """An issue as a past triage decision: its labels and week are fair to show."""
    body = case.body if full else case.body[:BODY_PREVIEW]
    return {
        "id": case.id,
        "title": case.title,
        "status": status_of(case),
        "labels": sorted(case.truth.labels),
        "milestone": case.truth.milestone,
        "body": body,
    }


def _corrupt(item: dict[str, Any]) -> dict[str, Any]:
    """Bad data the way real APIs serve it: truncated text, missing and wrong fields."""
    return {
        "id": item["id"],
        "title": item["title"][: len(item["title"]) // 2],
        "status": None,
        "labels": [],
        "milestone": "�",
        "body": item["body"][:40] + "��",
    }


class ToolError(Exception):
    """A tool failure the agent should see, the way a real API reports one."""


SEARCH_DESCRIPTION = (
    "Keyword search over this repo's existing issues. Returns up to 5 matches as JSON, "
    "each with its id, title, status, labels, milestone and the start of its body. "
    "Use specific words; there is no synonym matching."
)
GET_DESCRIPTION = 'Fetch one existing issue in full by id (for example "7"), as JSON.'


class IssueTools:
    """The two tools, free of any agent framework. Each framework wraps these methods.

    They read `corpus` (real issues) only. The issue being triaged is never visible.
    """

    def __init__(
        self,
        corpus: Sequence[TriageCase],
        *,
        target_id: str,
        injector: FailureInjector | None = None,
    ) -> None:
        self.target_id = target_id
        self.index = KeywordIndex(corpus)
        self.by_id = {c.id: c for c in corpus if c.id != target_id}
        self.inject = injector or FailureInjector()

    def search_issues(self, query: str) -> str:
        mode = self.inject.pick("search_issues")
        if mode == "timeout":
            raise ToolError("search_issues timed out after 30s")
        if mode == "empty":
            return "[]"
        if mode == "noise":
            hits = self.inject.shuffle(list(self.by_id.values()))[:5]
        else:
            hits = [case for case, _ in self.index.search(query, exclude=self.target_id)]
        items = [summarise(c) for c in hits]
        if mode == "bad-data":
            items = [_corrupt(i) for i in items]
        return json.dumps(items, ensure_ascii=False)

    def get_issue(self, issue_id: str) -> str:
        mode = self.inject.pick("get_issue")
        if mode == "timeout":
            raise ToolError("get_issue timed out after 30s")
        case = self.by_id.get(issue_id.lstrip("#"))
        if case is None or mode == "empty":
            raise ToolError(f"issue {issue_id} not found")
        if mode == "noise":
            case = self.inject.shuffle(list(self.by_id.values()))[0]
        item = summarise(case, full=True)
        return json.dumps(_corrupt(item) if mode == "bad-data" else item, ensure_ascii=False)


def make_tools(
    corpus: Sequence[TriageCase],
    *,
    target_id: str,
    injector: FailureInjector | None = None,
) -> list[BaseTool]:
    """The tools as LangChain tools, for the LangGraph agent."""
    core = IssueTools(corpus, target_id=target_id, injector=injector)

    @tool(description=SEARCH_DESCRIPTION)
    def search_issues(query: str) -> str:
        try:
            return core.search_issues(query)
        except ToolError as error:
            raise ToolException(str(error)) from error

    @tool(description=GET_DESCRIPTION)
    def get_issue(issue_id: str) -> str:
        try:
            return core.get_issue(issue_id)
        except ToolError as error:
            raise ToolException(str(error)) from error

    for t in (search_issues, get_issue):
        t.handle_tool_error = True  # the agent sees the error message, as with a real API
    return [search_issues, get_issue]
