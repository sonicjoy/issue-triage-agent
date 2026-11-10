"""The triager on the OpenAI Agents SDK, for comparison with graph.py (LangGraph).

Same dataset, prompts, tools and rules; a different shape:

    Runner.run(triage_agent)   the model drives everything, there are no nodes
      search_issues / get_issue ...            (at most 8 calls)
      apply_triage(decision)                   its approval check validates the decision:
        invalid -> runs at once, returns the broken rules to the model (one retry)
        valid   -> the run pauses with an interruption
    pause: snapshot the run, draft the comment (comment_writer agent), ask a human
    Runner.run(state)          approved -> apply_triage writes the dry-run file; rejected -> stop

The SDK can only pause for a human on a tool call, so applying is a tool the model calls
rather than a step after the model. Approval is checked before tool guardrails run, so
validation lives in the approval check: an invalid decision never reaches the human.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from agents import (
    Agent,
    FunctionTool,
    MaxTurnsExceeded,
    Model,
    ModelResponse,
    ModelSettings,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    RunContextWrapper,
    RunHooks,
    Runner,
    RunState,
    ToolsToFinalOutputResult,
    function_tool,
    set_trace_processors,
    set_tracing_disabled,
    set_tracing_export_api_key,
    trace,
)
from agents.strict_schema import ensure_strict_json_schema
from agents.tool import FunctionToolResult
from agents.tool_context import ToolContext
from agents.tracing import Span, Trace, TracingProcessor, get_trace_provider
from agents.tracing.processors import default_processor
from flight_recorder.llm import MissingCredentialsError, api_key_for, parse_spec
from flight_recorder.settings import Settings
from openai import AsyncOpenAI
from pydantic import ValidationError

from issue_triager.apply import write_dry_run
from issue_triager.dataset import Dataset, TriageCase
from issue_triager.decision import TriageDecision, decision_model
from issue_triager.graph import RESULT_PREVIEW, Usage
from issue_triager.prompts import COMMENT_SYSTEM, triage_request, triage_system
from issue_triager.tools import (
    GET_DESCRIPTION,
    SEARCH_DESCRIPTION,
    FailureInjector,
    IssueTools,
    ToolError,
)

MAX_TOOL_CALLS = 8  # search_issues + get_issue, as graph.py's ToolCallLimitMiddleware
MAX_ATTEMPTS = 2  # apply_triage calls: first try plus one retry after validation feedback
MAX_TURNS = MAX_TOOL_CALLS + MAX_ATTEMPTS + 2  # one turn is one model call
FINISH = (
    "When you have decided, call apply_triage with your decision. "
    "A maintainer reviews it before anything is applied."
)
REJECTED = "A maintainer rejected this triage. Nothing was applied. Stop."
STEP_OF_AGENT = {"triage_agent": "triage", "comment_writer": "comment"}

# Gemini and Claude through their OpenAI-compatible endpoints, with the SDK's own
# Chat Completions model: no extra dependency.
COMPATIBLE_BASE_URLS = {
    "google_genai": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "anthropic": "https://api.anthropic.com/v1/",
}

Review = Callable[[dict[str, Any]], bool]


# ---- Models ----------------------------------------------------------------------


@dataclass(frozen=True)
class SdkModels:
    triage: Model
    comment: Model
    triage_spec: str
    comment_spec: str


def sdk_model(spec: str, settings: Settings) -> Model:
    """An Agents SDK model from a `<provider>:<model>` spec, with the key from settings."""
    parsed = parse_spec(spec)
    key = api_key_for(parsed.provider, settings)
    if key is None:
        raise MissingCredentialsError(f"{parsed} needs an API key in the environment or .env")
    if parsed.provider == "openai":
        client = AsyncOpenAI(api_key=key.get_secret_value(), max_retries=2)
        return OpenAIResponsesModel(parsed.model, openai_client=client)
    client = AsyncOpenAI(
        api_key=key.get_secret_value(),
        base_url=COMPATIBLE_BASE_URLS[parsed.provider],
        max_retries=2,
    )
    return OpenAIChatCompletionsModel(parsed.model, openai_client=client)


# ---- Run context: what LangGraph keeps in graph state ------------------------------


@dataclass
class TriageContext:
    """Per-run data the tools read and write. The SDK never sends it to the model."""

    case: TriageCase
    dataset: Dataset
    tools: IssueTools
    schema: type[TriageDecision]
    models: SdkModels
    injector: FailureInjector
    runs_dir: Path
    tool_calls: int = 0
    attempts: int = 0
    decision: dict[str, Any] | None = None
    feedback: list[str] = field(default_factory=list[str])
    comment: str = ""
    approved: bool = False  # set between a human's approval and the apply_triage call it allows
    status: str | None = None
    output_path: str = ""
    steps: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    usage: list[Usage] = field(default_factory=list[Usage])

    def progress(self) -> dict[str, Any]:
        """The JSON-able part, stored in a paused run's snapshot.

        Without this the SDK deep-copies the whole context (model clients included).
        Resuming in another process would rebuild the rest from the dataset and the case id.
        """
        return {
            "case_id": self.case.id,
            "tool_calls": self.tool_calls,
            "attempts": self.attempts,
            "decision": self.decision,
            "steps": self.steps,
            "usage": self.usage,
        }


def _problems(c: TriageContext, params: dict[str, Any]) -> list[str]:
    try:
        decision = c.schema.model_validate(params)
    except ValidationError as error:
        return [f"The decision does not fit the schema: {error.errors()[0]['msg']}"]
    return c.dataset.problems(decision.to_truth(), target_id=c.case.id)


# ---- Tools -------------------------------------------------------------------------


def _show_error(ctx: RunContextWrapper[Any], error: Exception) -> str:
    """Show the model what went wrong, as graph.py's handle_tool_error=True does.

    The SDK's default hides the error ("An error occurred while running the tool").
    """
    return str(error)


def _issue_call(
    ctx: RunContextWrapper[TriageContext], tool: str, args: dict[str, Any],
    run: Callable[[IssueTools], str],
) -> str:  # fmt: skip
    c = ctx.context
    step: dict[str, Any] = {"attempt": c.attempts + 1, "tool": tool, "args": args}
    c.steps.append(step)
    c.tool_calls += 1
    try:
        if c.tool_calls > MAX_TOOL_CALLS:
            raise ToolError(
                f"Tool call limit of {MAX_TOOL_CALLS} reached. Decide now with apply_triage."
            )
        result = run(c.tools)
    except ToolError:
        step["status"] = "error"
        raise
    step |= {"status": "success", "result": result[:RESULT_PREVIEW]}
    return result


@function_tool(
    name_override="search_issues",
    description_override=SEARCH_DESCRIPTION,
    use_docstring_info=False,
    failure_error_function=_show_error,
)
def search_issues(ctx: RunContextWrapper[TriageContext], query: str) -> str:
    return _issue_call(ctx, "search_issues", {"query": query}, lambda t: t.search_issues(query))


@function_tool(
    name_override="get_issue",
    description_override=GET_DESCRIPTION,
    use_docstring_info=False,
    failure_error_function=_show_error,
)
def get_issue(ctx: RunContextWrapper[TriageContext], issue_id: str) -> str:
    return _issue_call(ctx, "get_issue", {"issue_id": issue_id}, lambda t: t.get_issue(issue_id))


async def _needs_review(
    ctx: RunContextWrapper[TriageContext], params: dict[str, Any], call_id: str
) -> bool:
    """Valid decisions pause for a human; invalid ones run at once and go back to the model."""
    return not _problems(ctx.context, params)


async def _apply(ctx: ToolContext[TriageContext], raw: str) -> str:
    c = ctx.context
    params: dict[str, Any] = json.loads(raw)
    if not c.approved:  # invalid, so no human saw it: count the attempt here
        c.attempts += 1
        c.steps.append({"attempt": c.attempts, "tool": "apply_triage", "args": params})
    c.approved = False
    c.feedback = _problems(c, params)
    if c.feedback:
        c.steps[-1]["status"] = "invalid"
        c.decision = params
        if c.attempts >= MAX_ATTEMPTS:
            c.status = "failed"
            return "Too many invalid decisions. Nothing was applied."
        rules = "\n".join(f"- {p}" for p in c.feedback)
        return f"Your decision broke these rules:\n{rules}\nDecide again."
    c.decision = c.schema.model_validate(params).model_dump()
    record = {
        "engine": "openai-agents",
        "repo": c.dataset.repo.repo,
        "case_id": c.case.id,
        "models": {"triage": c.models.triage_spec, "comment": c.models.comment_spec},
        "decision": c.decision,
        "comment": c.comment,
        "attempts": c.attempts,
        "steps": c.steps,
        "injected_failures": c.injector.injected,
        "usage": c.usage,
    }
    c.steps[-1]["status"] = "applied"
    c.output_path = write_dry_run(c.runs_dir, c.case.id, record).as_posix()
    c.status = "applied"
    return f"Applied (dry run) to {c.output_path}."


def apply_triage_tool(schema: type[TriageDecision]) -> FunctionTool:
    """Built per repo: its arguments are the repo's decision schema, enums included."""
    return FunctionTool(
        name="apply_triage",
        description="Apply your triage decision to the issue. Call once you have decided.",
        params_json_schema=ensure_strict_json_schema(schema.model_json_schema()),
        on_invoke_tool=_apply,
        needs_approval=_needs_review,
    )


def _settled(
    ctx: RunContextWrapper[TriageContext], results: list[FunctionToolResult]
) -> ToolsToFinalOutputResult:
    """End the run once apply_triage has applied, been rejected, or failed."""
    for result in results:
        if result.tool.name == "apply_triage" and result.output == REJECTED:
            ctx.context.status = "rejected"
    if ctx.context.status:
        return ToolsToFinalOutputResult(is_final_output=True, final_output=ctx.context.status)
    return ToolsToFinalOutputResult(is_final_output=False)


# ---- Agents ------------------------------------------------------------------------


def build_agents(
    dataset: Dataset, models: SdkModels, schema: type[TriageDecision]
) -> tuple[Agent[TriageContext], Agent[TriageContext]]:
    triage_agent = Agent[TriageContext](
        name="triage_agent",
        instructions=f"{triage_system(dataset.repo, dataset.rules)}\n\n{FINISH}",
        model=models.triage,
        tools=[search_issues, get_issue, apply_triage_tool(schema)],
        # Every turn must call a tool, so the run only ends through _settled or the turn limit.
        model_settings=ModelSettings(tool_choice="required"),
        reset_tool_choice=False,
        tool_use_behavior=_settled,
    )
    comment_writer = Agent[TriageContext](
        name="comment_writer", instructions=COMMENT_SYSTEM, model=models.comment
    )
    return triage_agent, comment_writer


class UsageHooks(RunHooks[TriageContext]):
    """Token usage per model call, in the same rows graph.py reports."""

    async def on_llm_end(
        self, context: RunContextWrapper[TriageContext], agent: Agent[TriageContext],
        response: ModelResponse,
    ) -> None:  # fmt: skip
        c = context.context
        step = STEP_OF_AGENT.get(agent.name, agent.name)
        spec = c.models.comment_spec if step == "comment" else c.models.triage_spec
        c.usage.append(
            Usage(
                step=step,
                model=spec,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        )


# ---- Tracing -----------------------------------------------------------------------


class TraceFile(TracingProcessor):
    """Writes the SDK's traces and spans to a local JSON Lines file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file: TextIO = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def _write(self, kind: str, item: Trace | Span[Any]) -> None:
        data = item.export()
        if data is not None:
            with self._lock:
                self._file.write(json.dumps({"event": kind, **data}, default=str) + "\n")

    def on_trace_start(self, trace: Trace) -> None:
        self._write("trace_start", trace)

    def on_trace_end(self, trace: Trace) -> None:
        self._write("trace_end", trace)

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        self._write("span", span)

    def shutdown(self) -> None:
        with self._lock:
            self._file.close()

    def force_flush(self) -> None:
        with self._lock:
            self._file.flush()


def _registered_processors() -> list[TracingProcessor]:
    """The SDK can add or replace processors but not list them, so read its private list.

    Returns [] if the SDK has moved it; then only our own processors are kept.
    """
    multi = getattr(get_trace_provider(), "_multi_processor", None)
    processors = getattr(multi, "_processors", None)
    return list(processors) if isinstance(processors, tuple | list) else []


def configure_tracing(path: Path, *, upload_key: str | None = None) -> TraceFile:
    """Keep traces local; also upload them to OpenAI only when given a key for it.

    Replaces the SDK's default upload to OpenAI and any earlier run's trace file, but keeps
    processors someone else registered, such as the OpenInference one `--record` adds.
    """
    local = TraceFile(path)
    uploader = default_processor()
    others = [
        p for p in _registered_processors() if p is not uploader and not isinstance(p, TraceFile)
    ]
    processors: list[TracingProcessor] = [local, *others]
    if upload_key:
        set_tracing_export_api_key(upload_key)
        processors.append(uploader)
    set_trace_processors(processors)
    set_tracing_disabled(False)
    return local


# ---- Run ---------------------------------------------------------------------------


async def _draft_comment(
    comment_writer: Agent[TriageContext], c: TriageContext, hooks: UsageHooks
) -> str:
    issue = c.case.agent_input()
    prompt = f"Issue: {issue.title}\n\nDecision: {json.dumps(c.decision)}"
    result = await Runner.run(comment_writer, prompt, context=c, hooks=hooks)
    return str(result.final_output).strip()


async def _triage(c: TriageContext, review: Review, trace_upload_key: str | None) -> None:
    triage_agent, comment_writer = build_agents(c.dataset, c.models, c.schema)
    hooks = UsageHooks()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    tracer = configure_tracing(
        c.runs_dir / f"trace-{c.case.id}-{stamp}.jsonl", upload_key=trace_upload_key
    )
    try:
        with trace("issue_triager", metadata={"repo": c.dataset.repo.repo, "case": c.case.id}):
            prompt = triage_request(c.case.agent_input())
            result = await Runner.run(
                triage_agent, prompt, context=c, max_turns=MAX_TURNS, hooks=hooks
            )
            while result.interruptions:  # only apply_triage asks for approval
                # A pause is a JSON snapshot: it could be stored and resumed in another process.
                snapshot = result.to_state().to_string(
                    context_serializer=lambda context: context.progress()
                )
                state = await RunState.from_string(
                    triage_agent, snapshot, context_override=RunContextWrapper(c)
                )
                for item in state.get_interruptions():
                    c.attempts += 1
                    c.decision = json.loads(item.arguments or "{}")
                    step = {
                        "attempt": c.attempts,
                        "tool": "apply_triage",
                        "args": c.decision,
                        "status": "awaiting review",
                    }
                    c.steps.append(step)
                    c.comment = await _draft_comment(comment_writer, c, hooks)
                    view = {"decision": c.decision, "comment": c.comment, "steps": c.steps}
                    c.approved = review(view)
                    step["status"] = "approved" if c.approved else "rejected"
                    if c.approved:
                        state.approve(item)
                    else:
                        state.reject(item, rejection_message=REJECTED)
                result = await Runner.run(triage_agent, state, max_turns=MAX_TURNS, hooks=hooks)
    except MaxTurnsExceeded:
        c.status = "failed"
        c.feedback = [f"No decision within {MAX_TURNS} model calls."]
    finally:
        tracer.force_flush()
    c.status = c.status or "failed"


def triage(
    case: TriageCase,
    dataset: Dataset,
    models: SdkModels,
    *,
    review: Review,
    injector: FailureInjector | None = None,
    runs_dir: Path = Path("runs"),
    trace_upload_key: str | None = None,
) -> dict[str, Any]:
    """Triage `case`, asking `review` before applying. Returns the same keys as graph.py's state."""
    injector = injector or FailureInjector()
    c = TriageContext(
        case=case,
        dataset=dataset,
        tools=IssueTools(list(dataset.corpus.values()), target_id=case.id, injector=injector),
        schema=decision_model(dataset.repo),
        models=models,
        injector=injector,
        runs_dir=runs_dir,
    )
    asyncio.run(_triage(c, review, trace_upload_key))
    return {
        "decision": c.decision,
        "comment": c.comment,
        "attempts": c.attempts,
        "feedback": c.feedback,
        "steps": c.steps,
        "usage": c.usage,
        "status": c.status,
        "output_path": c.output_path,
    }
