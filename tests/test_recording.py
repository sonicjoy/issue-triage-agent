"""`triager run --record`: traces go to an OTLP endpoint only when asked, for both engines."""

from __future__ import annotations

import builtins
import socket
import threading
from collections.abc import Generator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from agents import add_trace_processor, set_trace_processors
from agents.tracing import TracingProcessor
from agents.tracing.processors import default_processor
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from test_agents_sdk import apply, search
from test_agents_sdk import models as sdk_models
from test_triager import call, decide
from test_triager import models as langgraph_models
from typer.testing import CliRunner

from issue_triager import agents_sdk, cli, recording
from issue_triager.recording import Target, endpoint_from, shown, traces_url


class Collector:
    """A tiny OTLP/HTTP endpoint that keeps what it's sent."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, bytes]] = []
        collector = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                collector.requests.append((self.path, self.headers["Content-Type"], body))
                self.send_response(200)
                self.send_header("Content-Type", "application/x-protobuf")
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def received(self) -> Iterator[tuple[Any, Any]]:
        """(resource, span) protobufs for every span received."""
        for _, _, body in self.requests:
            for resource_spans in ExportTraceServiceRequest.FromString(body).resource_spans:
                for scope in resource_spans.scope_spans:
                    for span in scope.spans:
                        yield resource_spans.resource, span

    def spans(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """(resource attributes, span attributes) for every span received."""
        return [
            (
                {a.key: a.value.string_value for a in resource.attributes},
                {a.key: a.value.string_value for a in span.attributes} | {"name": span.name},
            )
            for resource, span in self.received()
        ]

    def raw_spans(self) -> list[Any]:
        return [span for _, span in self.received()]


@pytest.fixture
def collector() -> Generator[Collector]:
    server = Collector()
    yield server
    server.server.shutdown()


@pytest.fixture(autouse=True)
def restore_agents_tracing() -> Generator[None]:
    yield
    set_trace_processors([default_processor()])


def test_endpoint_option_then_environment_then_off(clean_env: pytest.MonkeyPatch) -> None:
    assert endpoint_from(None) == Target(None)
    clean_env.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://env:4318")
    assert endpoint_from(None) == Target("http://env:4318/v1/traces")
    assert endpoint_from("http://option:4318") == Target("http://option:4318/v1/traces")


BASE = {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://b:4318"}


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        # The traces-specific endpoint is used as is, and wins over the base one.
        (BASE | {"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://t:9/x"}, Target("http://t:9/x")),
        (BASE | {"OTEL_SDK_DISABLED": "TRUE"}, Target(None)),
        (BASE | {"OTEL_TRACES_EXPORTER": "none"}, Target(None)),
        (BASE | {"OTEL_TRACES_EXPORTER": "console, otlp"}, Target("http://b:4318/v1/traces")),
        (
            BASE | {"OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf"},
            Target("http://b:4318/v1/traces"),
        ),
        # The traces-specific protocol wins over the general one.
        (
            BASE
            | {
                "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
                "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
            },
            Target("http://b:4318/v1/traces"),
        ),
    ],
)
def test_endpoint_follows_the_standard_otel_variables(
    environ: dict[str, str], expected: Target
) -> None:
    assert endpoint_from(None, environ) == expected


def test_grpc_in_the_environment_turns_recording_off_with_a_note() -> None:
    environ = BASE | {"OTEL_EXPORTER_OTLP_PROTOCOL": "grpc"}
    target = endpoint_from(None, environ)

    assert target.url is None
    assert target.note is not None and "OTEL_EXPORTER_OTLP_PROTOCOL=grpc" in target.note
    # --record is explicit, so it wins over every environment setting.
    assert endpoint_from("http://x:4318", environ | {"OTEL_SDK_DISABLED": "true"}).url


def test_shown_hides_credentials_and_query() -> None:
    assert shown("https://user:tok@host:4318/v1/traces?key=s#f") == "https://host:4318/v1/traces"
    assert shown("http://[::1]:4318/v1/traces") == "http://[::1]:4318/v1/traces"


@pytest.mark.parametrize(
    "endpoint",
    ["http://127.0.0.1:4318", "http://127.0.0.1:4318/", "http://127.0.0.1:4318/v1/traces"],
)
def test_traces_url(endpoint: str) -> None:
    assert traces_url(endpoint) == "http://127.0.0.1:4318/v1/traces"


def test_traces_url_keeps_a_query_after_the_path() -> None:
    assert traces_url("http://h:4318/otel?key=k") == "http://h:4318/otel/v1/traces?key=k"


def run_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine: str, *extra: str) -> Any:
    if engine == "langgraph":
        fake = langgraph_models(iter([call("search_issues", {"query": "evals"}), decide()]))
        monkeypatch.setattr(cli, "resolve_models", lambda *a, **k: fake)
    else:
        scripted = sdk_models(search(), apply())
        monkeypatch.setattr(cli, "resolve_sdk_models", lambda *a, **k: scripted)
    return CliRunner().invoke(
        cli.app,
        ["run", "--issue", "19", "--engine", engine, "--yes", "--runs-dir", str(tmp_path), *extra],
    )


def test_nothing_is_sent_unless_asked(
    clean_env: pytest.MonkeyPatch, tmp_path: Path, collector: Collector
) -> None:
    result = run_cli(clean_env, tmp_path, "langgraph")

    assert result.exit_code == 0, result.output
    assert "record:" not in result.output
    assert collector.requests == []


@pytest.mark.parametrize(
    ("engine", "expected_kinds"),
    [
        ("langgraph", {"CHAIN", "LLM", "TOOL"}),
        # The SDK's ScriptedModel makes no model call, so the SDK opens no response span to
        # turn into an LLM span; real runs have them (see the PR's evidence).
        ("openai-agents", {"AGENT", "TOOL"}),
    ],
)
def test_record_sends_openinference_spans(
    clean_env: pytest.MonkeyPatch,
    tmp_path: Path,
    collector: Collector,
    engine: str,
    expected_kinds: set[str],
) -> None:
    result = run_cli(clean_env, tmp_path, engine, "--record", collector.url)

    assert result.exit_code == 0, result.output
    assert f"record:  traces to {collector.url}/v1/traces" in result.output
    assert {path for path, _, _ in collector.requests} == {"/v1/traces"}
    assert {kind for _, kind, _ in collector.requests} == {"application/x-protobuf"}
    spans = collector.spans()
    assert {r["service.name"] for r, _ in spans} == {"issue-triager"}
    assert {r["triager.engine"] for r, _ in spans} == {engine}
    assert expected_kinds <= {a.get("openinference.span.kind") for _, a in spans}


@pytest.mark.parametrize(
    ("engine", "missing"),
    [
        ("langgraph", "opentelemetry.exporter"),
        ("langgraph", "openinference.instrumentation.langchain"),
        ("openai-agents", "openinference.instrumentation.openai_agents"),
    ],
)
def test_missing_extra_is_explained(
    clean_env: pytest.MonkeyPatch, tmp_path: Path, collector: Collector, engine: str, missing: str
) -> None:
    real_import = builtins.__import__

    def without(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(missing):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    clean_env.setattr(builtins, "__import__", without)
    result = run_cli(clean_env, tmp_path, engine, "--record", collector.url)

    assert result.exit_code == 2
    assert "uv sync --extra record" in result.output


class Other(TracingProcessor):
    def on_trace_start(self, trace: Any) -> None: ...
    def on_trace_end(self, trace: Any) -> None: ...
    def on_span_start(self, span: Any) -> None: ...
    def on_span_end(self, span: Any) -> None: ...
    def shutdown(self) -> None: ...
    def force_flush(self) -> None: ...


def test_local_trace_keeps_processors_others_registered(tmp_path: Path) -> None:
    """--record's OpenInference processor must survive the per-run configure_tracing."""
    other = Other()
    add_trace_processor(other)  # alongside the SDK's default OpenAI uploader
    first = agents_sdk.configure_tracing(tmp_path / "a.jsonl")

    second = agents_sdk.configure_tracing(tmp_path / "b.jsonl")

    registered = agents_sdk._registered_processors()  # pyright: ignore[reportPrivateUsage]
    assert other in registered and second in registered
    assert first not in registered  # the earlier run's file is dropped
    assert default_processor() not in registered  # no upload to OpenAI


@pytest.mark.parametrize("engine", ["langgraph", "openai-agents"])
def test_one_triage_is_one_trace_under_a_root_span(
    clean_env: pytest.MonkeyPatch, tmp_path: Path, collector: Collector, engine: str
) -> None:
    """LangGraph resumes after the review in a second graph call: still the same trace."""
    result = run_cli(clean_env, tmp_path, engine, "--record", collector.url)

    assert result.exit_code == 0, result.output
    spans = collector.raw_spans()
    assert len({s.trace_id for s in spans}) == 1
    [root] = [s for s in spans if not s.parent_span_id]
    assert root.name == "triage sonicjoy/agent-flight-recorder#19"
    assert root.status.code == root.status.STATUS_CODE_OK


def test_a_failed_triage_marks_the_root_span_as_an_error(
    clean_env: pytest.MonkeyPatch, tmp_path: Path, collector: Collector
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("provider unavailable")

    clean_env.setattr(cli, "run_langgraph", broken)
    result = run_cli(clean_env, tmp_path, "langgraph", "--record", collector.url)

    assert result.exit_code == 1
    [root] = [s for s in collector.raw_spans() if not s.parent_span_id]
    assert root.status.code == root.status.STATUS_CODE_ERROR
    assert any(event.name == "exception" for event in root.events)


def closed_port_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def test_an_unreachable_endpoint_is_reported_not_silent(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clean_env.setattr(recording, "EXPORT_TIMEOUT", 0.5)
    result = run_cli(clean_env, tmp_path, "langgraph", "--record", closed_port_url())

    assert result.exit_code == 0, result.output  # the triage itself worked
    assert "batch(es) of spans didn't reach" in result.output
    assert "status:  applied" in result.output


def test_a_failure_while_stopping_doesnt_hide_the_result(
    clean_env: pytest.MonkeyPatch, tmp_path: Path, collector: Collector
) -> None:
    from opentelemetry.sdk.trace import TracerProvider

    def broken(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("flush broke")

    clean_env.setattr(TracerProvider, "force_flush", broken)
    result = run_cli(clean_env, tmp_path, "langgraph", "--record", collector.url)

    assert result.exit_code == 0, result.output
    assert "record:  stopping: RuntimeError: flush broke" in result.output
    assert "status:  applied" in result.output


def test_credentials_in_the_endpoint_are_not_printed(
    clean_env: pytest.MonkeyPatch, tmp_path: Path, collector: Collector
) -> None:
    secret_url = collector.url.replace("http://", "http://user:s3cret@") + "?token=s3cret"
    result = run_cli(clean_env, tmp_path, "langgraph", "--record", secret_url)

    assert result.exit_code == 0, result.output
    assert "s3cret" not in result.output
    assert f"record:  traces to {collector.url}/v1/traces" in result.output
