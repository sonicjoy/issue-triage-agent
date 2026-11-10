"""Send the triager's traces to an OTLP endpoint, such as Agent Flight Recorder.

Off unless configured: `triager run --record URL`, or the standard OTel environment
variables (OTEL_EXPORTER_OTLP_ENDPOINT and friends). Needs the `record` extra, which `uv sync`
installs with the dev tools; for an install without them, `uv sync --extra record`.

Uses the OpenInference instrumentors, as any outside agent would: LangChain/LangGraph for
`--engine langgraph`, the OpenAI Agents SDK for `--engine openai-agents`. They record prompts
and outputs; the recorder redacts secrets before storing anything.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NamedTuple
from urllib.parse import urlsplit, urlunsplit

ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
TRACES_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
PROTOCOL_ENVS = ("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "OTEL_EXPORTER_OTLP_PROTOCOL")
TIMEOUT_ENVS = ("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT", "OTEL_EXPORTER_OTLP_TIMEOUT")
TRACES_PATH = "/v1/traces"
SERVICE = "issue-triager"
EXPORT_TIMEOUT = 5.0  # seconds per batch, retries included: bounds the wait at exit


class RecordingUnavailable(RuntimeError):
    """The `record` extra isn't installed."""


class Target(NamedTuple):
    url: str | None  # where to send traces; None: recording off
    note: str | None = None  # why an endpoint in the environment was ignored


def endpoint_from(option: str | None, environ: Mapping[str, str] = os.environ) -> Target:
    """--record if given; otherwise the standard OTel variables, as an OTel SDK reads them.

    OTEL_SDK_DISABLED or an OTEL_TRACES_EXPORTER without `otlp` turn it off.
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is used as is; OTEL_EXPORTER_OTLP_ENDPOINT is a base
    URL. Only http/protobuf is sent, so a gRPC setting turns recording off, with a note.
    """
    if option:
        return Target(traces_url(option))

    def env(name: str) -> str:
        return environ.get(name, "").strip()

    exporters = {e.strip().lower() for e in env("OTEL_TRACES_EXPORTER").split(",") if e.strip()}
    if env("OTEL_SDK_DISABLED").lower() == "true" or (exporters and "otlp" not in exporters):
        return Target(None)
    url = env(TRACES_ENDPOINT_ENV) or (traces_url(base) if (base := env(ENDPOINT_ENV)) else "")
    if not url:
        return Target(None)
    for name in PROTOCOL_ENVS:
        if protocol := env(name):
            if protocol != "http/protobuf":
                note = f"off: {name}={protocol}, and the triager only sends http/protobuf"
                return Target(None, f"{note}; pass --record with an OTLP/HTTP endpoint instead")
            break  # the traces-specific setting wins over the general one
    return Target(url)


def traces_url(endpoint: str) -> str:
    """OTEL_EXPORTER_OTLP_ENDPOINT is a base URL; the exporter needs the traces path."""
    parts = urlsplit(endpoint)
    path = parts.path.rstrip("/")
    return urlunsplit(
        parts._replace(path=path if path.endswith(TRACES_PATH) else path + TRACES_PATH)
    )


def shown(url: str) -> str:
    """The URL without credentials or query, safe to print."""
    parts = urlsplit(url)
    return urlunsplit(parts._replace(netloc=parts.netloc.rpartition("@")[2], query="", fragment=""))


@dataclass
class Recording:
    url: str
    provider: Any  # opentelemetry.sdk.trace.TracerProvider
    instrumentor: Any
    exporter: Any  # counts the batches that weren't delivered

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

    def stop(self) -> str | None:
        """Send what's buffered, then stop. Spans are batched, so this matters at exit.

        Never raises, so a recording problem can't hide the triage's own result. Returns
        what went wrong, or None. OTel itself only logs a failed export.
        """
        problems: list[str] = []
        for step in (
            self.provider.force_flush,
            self.instrumentor.uninstrument,
            self.provider.shutdown,
        ):
            try:
                step()
            except Exception as error:
                problems.append(f"stopping: {type(error).__name__}: {error}")
        if self.exporter.failed:
            lost = f"{self.exporter.failed} batch(es) of spans didn't reach {shown(self.url)}"
            problems.insert(0, f"{lost}; is it running?")
        return "; ".join(problems) or None


def start_recording(url: str, *, engine: str) -> Recording:
    """Instrument `engine` and send its spans to the OTLP/HTTP traces `url`."""
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult

        instrumentor: Any
        if engine == "langgraph":
            from openinference.instrumentation.langchain import LangChainInstrumentor

            instrumentor = LangChainInstrumentor()
        else:
            from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

            instrumentor = OpenAIAgentsInstrumentor()
    except ImportError as error:
        raise RecordingUnavailable(
            "recording needs the `record` extra: uv sync --extra record"
        ) from error

    class Exporter(OTLPSpanExporter):
        """Counts failed batches, which the SDK only logs."""

        failed = 0

        def export(self, spans: Any) -> SpanExportResult:
            result = super().export(spans)
            if result is not SpanExportResult.SUCCESS:
                self.failed += 1
            return result

    timeout_set = any(os.environ.get(name) for name in TIMEOUT_ENVS)
    exporter = Exporter(endpoint=url, timeout=None if timeout_set else EXPORT_TIMEOUT)
    provider = TracerProvider(
        resource=Resource.create({"service.name": SERVICE, "triager.engine": engine})
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    instrumentor.instrument(tracer_provider=provider)
    return Recording(url, provider, instrumentor, exporter)
