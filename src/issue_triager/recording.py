"""Send the triager's traces to an OTLP endpoint, such as Agent Flight Recorder.

Off unless configured: `triager run --record URL`, or the standard OTEL_EXPORTER_OTLP_ENDPOINT.
Needs the optional extra: `uv sync --extra record`.

Uses the OpenInference instrumentors, as any outside agent would: LangChain/LangGraph for
`--engine langgraph`, the OpenAI Agents SDK for `--engine openai-agents`. They record prompts
and outputs; the recorder redacts secrets before storing anything.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
TRACES_PATH = "/v1/traces"
SERVICE = "issue-triager"


class RecordingUnavailable(RuntimeError):
    """The `record` extra isn't installed."""


def endpoint_from(option: str | None) -> str | None:
    """The --record value, else OTEL_EXPORTER_OTLP_ENDPOINT, else None (recording off)."""
    return option or os.environ.get(ENDPOINT_ENV) or None


def traces_url(endpoint: str) -> str:
    """OTEL_EXPORTER_OTLP_ENDPOINT is a base URL; the exporter needs the traces path."""
    base = endpoint.rstrip("/")
    return base if base.endswith(TRACES_PATH) else base + TRACES_PATH


@dataclass
class Recording:
    url: str
    provider: Any  # opentelemetry.sdk.trace.TracerProvider
    instrumentor: Any

    @contextmanager
    def triage_span(self, name: str, attributes: dict[str, str]) -> Iterator[Any]:
        """One root span around a whole triage.

        Without it, LangGraph's resume after the human review starts a second trace (a
        second run in the recorder), and a failed SDK run looks fine: its own root span
        doesn't see the exception. An exception here marks the span as an error.
        """
        tracer = self.provider.get_tracer("issue_triager")
        root = {"openinference.span.kind": "AGENT", "agent.name": SERVICE} | attributes
        with tracer.start_as_current_span(name, attributes=root) as span:
            yield span

    def finish(self, span: Any, *, failed: str | None = None) -> None:
        """Say how the triage ended: OK (applied or rejected by the reviewer), or an error.

        OTel leaves a span's status unset unless told; the triager knows the outcome.
        An exception is recorded as an error by the span itself.
        """
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, failed) if failed else Status(StatusCode.OK))

    def stop(self) -> None:
        """Send what's buffered, then stop. Spans are batched, so this matters at exit."""
        self.provider.force_flush()
        self.instrumentor.uninstrument()
        self.provider.shutdown()


def start_recording(endpoint: str, *, engine: str) -> Recording:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as error:
        raise RecordingUnavailable(
            "recording needs the `record` extra: uv sync --extra record"
        ) from error

    url = traces_url(endpoint)
    provider = TracerProvider(
        resource=Resource.create({"service.name": SERVICE, "triager.engine": engine})
    )
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=url)))
    if engine == "langgraph":
        from openinference.instrumentation.langchain import LangChainInstrumentor

        instrumentor: Any = LangChainInstrumentor()
    else:
        from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

        instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    return Recording(url, provider, instrumentor)
