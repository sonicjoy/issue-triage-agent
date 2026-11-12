"""Triage GitHub issues with an AI agent (dry run: nothing is written to GitHub)."""

from __future__ import annotations

import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from flight_recorder.llm import (
    MissingCredentialsError,
    ModelSpecError,
    api_key_for,
    chat_model,
)
from flight_recorder.pricing import cost_usd
from flight_recorder.settings import Settings
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from issue_triager import agents_sdk
from issue_triager.dataset import (
    DATASETS_DIR,
    DEFAULT_DATASET,
    Dataset,
    GroundTruth,
    TriageCase,
    dataset_dir_name,
    load_dataset,
)
from issue_triager.export import export as export_dataset
from issue_triager.grading import grade_attributes, grade_fields
from issue_triager.graph import Models, TriageState, Usage, build_graph
from issue_triager.recording import (
    Recording,
    RecordingUnavailable,
    endpoint_from,
    shown,
    start_recording,
)
from issue_triager.tools import FAILURE_MODES, FailureInjector, FailureMode

app = typer.Typer(add_completion=False, help=__doc__)
ENGINES = ("langgraph", "openai-agents")


def model_specs(
    model: str | None, comment_model: str | None, *, cheap: bool, settings: Settings
) -> tuple[str, str]:
    """--model sets both steps unless --comment-model is given; otherwise use settings."""
    triage_spec = model or (settings.llm_cheap_model if cheap else settings.llm_model)
    comment_spec = comment_model or model or settings.llm_cheap_model
    return triage_spec, comment_spec


def resolve_sdk_models(
    model: str | None, comment_model: str | None, *, cheap: bool, settings: Settings
) -> agents_sdk.SdkModels:
    triage_spec, comment_spec = model_specs(model, comment_model, cheap=cheap, settings=settings)
    return agents_sdk.SdkModels(
        triage=agents_sdk.sdk_model(triage_spec, settings),
        comment=agents_sdk.sdk_model(comment_spec, settings),
        triage_spec=triage_spec,
        comment_spec=comment_spec,
    )


def resolve_models(
    model: str | None, comment_model: str | None, *, cheap: bool, settings: Settings
) -> Models:
    triage_spec, comment_spec = model_specs(model, comment_model, cheap=cheap, settings=settings)
    return Models(
        triage=chat_model(triage_spec, settings=settings),
        comment=chat_model(comment_spec, settings=settings),
        triage_spec=triage_spec,
        comment_spec=comment_spec,
    )


def parse_inject(value: str) -> list[FailureMode]:
    modes = [m.strip() for m in value.split(",") if m.strip()]
    unknown = [m for m in modes if m not in FAILURE_MODES]
    if unknown:
        raise typer.BadParameter(f"unknown {unknown}; choose from {', '.join(FAILURE_MODES)}")
    return cast(list[FailureMode], modes)


def grade(decision: dict[str, Any], truth: GroundTruth) -> str:
    """Compare with the ground truth the agent never saw."""
    marks = grade_fields(decision, truth)
    return "; ".join(
        f"{name} {'ok' if ok else f'WRONG (want {getattr(truth, name)})'}"
        for name, ok in marks.items()
    )


def usage_cost(usage: list[Usage]) -> float | None:
    """The run's cost from its token usage; None if a model isn't priced."""
    total = 0.0
    for row in usage:
        cost = cost_usd(row["model"], row["input_tokens"], row["output_tokens"])
        if cost is None:
            return None
        total += cost
    return total


def report_usage(usage: list[Usage]) -> str:
    total = usage_cost(usage)
    tokens_in = sum(r["input_tokens"] for r in usage)
    tokens_out = sum(r["output_tokens"] for r in usage)
    cost_text = f"${total:.6f}" if total is not None else "unknown (unpriced model)"
    return f"tokens:  {tokens_in} in / {tokens_out} out   cost: {cost_text}"


def show_review(state: dict[str, Any], case: TriageCase) -> None:
    for step in state.get("steps", []):
        status = step.get("status", "?")
        typer.echo(f"  [{step['attempt']}] {step['tool']}({step['args']}) -> {status}")
    decision = state.get("decision") or {}
    typer.echo(f"decision: {decision}")
    typer.echo(f"vs truth: {grade(decision, case.truth)}")
    typer.echo(f"comment:\n{state.get('comment', '')}\n")


def finish_root(
    recording: Recording | None, span: Any, state: dict[str, Any], case: TriageCase, model: str
) -> dict[str, str]:
    """Put the grade on the run's root span and say how it ended; returns the grade."""
    grades = grade_attributes(state, case)
    if recording is not None and span is not None:
        span.set_attributes(grades | {"triager.triage_model": model})
        failed = state.get("status") == "failed"
        recording.finish(span, failed="failed validation" if failed else None)
    return grades


def ask(state: dict[str, Any], case: TriageCase, *, yes: bool) -> bool:
    show_review(state, case)
    return yes or typer.confirm("Apply (dry run: writes a local file)?", default=False)


def run_langgraph(
    case: TriageCase, data: Dataset, models: Models, injector: FailureInjector, *,
    yes: bool, runs_dir: Path,
) -> dict[str, Any]:  # fmt: skip
    graph = build_graph(
        case, data, models, checkpointer=InMemorySaver(), injector=injector, runs_dir=runs_dir
    )
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = graph.invoke(cast(TriageState, {"case_id": case.id}), config)
    if "__interrupt__" in state:
        state = graph.invoke(Command(resume=ask(dict(state), case, yes=yes)), config)
    return dict(state)


@app.command()
def run(
    issue: Annotated[str, typer.Option(help='Issue number ("7") or synthetic case id ("s53")')],
    dataset: Annotated[
        Path, typer.Option(help="Dataset folder from `triager export`")
    ] = DEFAULT_DATASET,
    engine: Annotated[str, typer.Option(help=f"Agent version: {' or '.join(ENGINES)}")] = (
        "langgraph"
    ),
    model: Annotated[str | None, typer.Option(help="<provider>:<model> for both steps")] = None,
    comment_model: Annotated[str | None, typer.Option(help="Model for the comment")] = None,
    cheap: Annotated[bool, typer.Option(help="Triage with the cheap model")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Approve without asking")] = False,
    inject: Annotated[str, typer.Option(help=f"Tool failures: {','.join(FAILURE_MODES)}")] = "",
    rate: Annotated[float, typer.Option(min=0, max=1, help="Chance per tool call")] = 0.3,
    seed: Annotated[int | None, typer.Option(help="Seed for repeatable injection")] = None,
    runs_dir: Annotated[Path, typer.Option(help="Where 'apply' writes")] = Path("runs"),
    openai_trace: Annotated[
        bool, typer.Option(help="openai-agents only: also upload the trace to OpenAI")
    ] = False,
    record: Annotated[
        str | None,
        typer.Option(
            help="Send traces to this OTLP endpoint, e.g. http://127.0.0.1:4318 (Agent Flight "
            "Recorder) [default: $OTEL_EXPORTER_OTLP_ENDPOINT; off if neither]"
        ),
    ] = None,
) -> None:
    """Triage one issue from a dataset end to end."""
    if engine not in ENGINES:
        typer.echo(f"unknown engine {engine!r}; choose from {', '.join(ENGINES)}")
        raise typer.Exit(2)
    if not (dataset / "issues.json").exists():
        typer.echo(f"no dataset at {dataset}; create one with `triager export OWNER/NAME`")
        raise typer.Exit(2)
    data = load_dataset(dataset)
    case = data.by_id.get(issue.lstrip("#"))
    if case is None:
        typer.echo(f"no issue {issue!r} in {data.repo.repo} ({len(data.real)} real issues)")
        raise typer.Exit(2)
    settings = Settings()
    upload_key = api_key_for("openai", settings) if openai_trace else None
    if openai_trace and upload_key is None:
        typer.echo("--openai-trace needs OPENAI_API_KEY in the environment or .env")
        raise typer.Exit(2)
    try:
        if engine == "langgraph":
            models: Models | agents_sdk.SdkModels = resolve_models(
                model, comment_model, cheap=cheap, settings=settings
            )
        else:
            models = resolve_sdk_models(model, comment_model, cheap=cheap, settings=settings)
    except (ModelSpecError, MissingCredentialsError) as error:
        typer.echo(str(error))
        raise typer.Exit(2) from error

    injector = FailureInjector(parse_inject(inject), rate=rate, seed=seed)
    typer.echo(f"triaging {data.repo.repo}#{case.id} ({case.source}): {case.title}")
    typer.echo(f"engine:  {engine}")
    typer.echo(f"models:  triage {models.triage_spec}, comment {models.comment_spec}")

    recording: Recording | None = None
    target = endpoint_from(record)
    if target.note:
        typer.echo(f"record:  {target.note}")
    if target.url:
        try:
            recording = start_recording(target.url, engine=engine)
        except RecordingUnavailable as error:
            typer.echo(f"error: {error}")
            raise typer.Exit(2) from error
        typer.echo(f"record:  traces to {shown(recording.url)}")

    started = time.perf_counter()
    root = (
        recording.triage_span(
            f"triage {data.repo.repo}#{case.id}",
            {"triager.repo": data.repo.repo, "triager.case": case.id, "triager.engine": engine},
        )
        if recording
        else nullcontext()
    )
    try:
        with root as span:
            if isinstance(models, Models):
                state = run_langgraph(case, data, models, injector, yes=yes, runs_dir=runs_dir)
            else:
                state = agents_sdk.triage(
                    case, data, models, review=lambda s: ask(s, case, yes=yes),
                    injector=injector, runs_dir=runs_dir,
                    trace_upload_key=upload_key.get_secret_value() if upload_key else None,
                )  # fmt: skip
            failed = state.get("status") == "failed"
            finish_root(recording, span, state, case, models.triage_spec)
    except Exception as error:  # provider and network errors: one clear line
        first_line = str(error).splitlines()[0] if str(error) else ""
        typer.echo(f"error: {type(error).__name__}: {first_line}")
        raise typer.Exit(1) from error
    finally:
        if recording and (problem := recording.stop()):  # flushes buffered spans
            typer.echo(f"record:  {problem}")

    if failed:
        show_review(state, case)
        typer.echo(
            f"failed validation after {state.get('attempts')} attempts: {state.get('feedback')}"
        )
    if injector.injected:
        typer.echo(f"injected: {', '.join(injector.injected)}")
    typer.echo(report_usage(state.get("usage", [])))
    typer.echo(f"status:  {state.get('status')}  {state.get('output_path', '')}".rstrip())
    typer.echo(f"latency: {time.perf_counter() - started:.1f}s")
    if failed:
        raise typer.Exit(1)


@app.command()
def export(
    repo: Annotated[str, typer.Argument(help="GitHub repo as OWNER/NAME")],
    out: Annotated[
        Path | None, typer.Option(help="Dataset folder [default: ./datasets/OWNER__NAME]")
    ] = None,
    limit: Annotated[int, typer.Option(help="Most issues to fetch")] = 1000,
) -> None:
    """Export a repo's issues, labels and milestones to a dataset (read-only, uses `gh`)."""
    if repo.count("/") != 1:
        typer.echo("repo must be OWNER/NAME")
        raise typer.Exit(2)
    target = out or Path("datasets") / dataset_dir_name(repo)
    issues = export_dataset(repo, target, limit=limit).issues
    labelled = sum(1 for c in issues if c.truth.labels)
    dupes = [f"#{c.id} -> #{c.truth.duplicate_of}" for c in issues if c.truth.duplicate_of]
    typer.echo(f"wrote {len(issues)} issues ({labelled} labelled) to {target.as_posix()}")
    typer.echo(f"duplicates: {', '.join(dupes) or 'none'}")
    if target.resolve().is_relative_to(DATASETS_DIR.resolve()):
        typer.echo("(bundled dataset: commit the change)")


@app.command()
def batch(
    plan: Annotated[Path, typer.Argument(help="Plan file (YAML): groups of runs")],
    dataset: Annotated[
        Path, typer.Option(help="Dataset folder from `triager export`")
    ] = DEFAULT_DATASET,
    manifest: Annotated[
        Path | None,
        typer.Option(help="Where each run is written [default: runs/batch-<plan name>.jsonl]"),
    ] = None,
    budget: Annotated[
        float | None, typer.Option(help="Stop at this spend in US dollars [default: the plan's]")
    ] = None,
    runs_dir: Annotated[Path, typer.Option(help="Where 'apply' writes")] = Path("runs"),
    record: Annotated[
        str | None,
        typer.Option(help="OTLP endpoint for traces [default: $OTEL_EXPORTER_OTLP_ENDPOINT]"),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="List the runs; run none")] = False,
) -> None:
    """Run a plan's matrix of triages (LangGraph engine), approving each, and record them.

    Resumable: runs already in the manifest are skipped; ones that ended in an error are tried
    again. Stops cleanly at the spend cap. Rate-limited calls back off and retry.
    """
    from issue_triager.batch import (
        Job,
        Outcome,
        PlanError,
        RunError,
        check_budget,
        load_plan,
        plan_jobs,
        run_batch,
    )

    try:
        loaded = load_plan(plan)
    except (OSError, PlanError) as error:
        typer.echo(f"error: {error}")
        raise typer.Exit(2) from error
    if not (dataset / "issues.json").exists():
        typer.echo(f"no dataset at {dataset}; create one with `triager export OWNER/NAME`")
        raise typer.Exit(2)
    data = load_dataset(dataset)
    jobs = plan_jobs(loaded, data)
    cap = budget if budget is not None else loaded.budget_usd
    if budget is not None:
        try:
            check_budget(loaded.groups, budget)
        except PlanError as error:
            typer.echo(f"error: {error}")
            raise typer.Exit(2) from error
    target_manifest = manifest or runs_dir / f"batch-{plan.stem}.jsonl"
    by_group: dict[str, int] = {}
    for job in jobs:
        by_group[job.group.name] = by_group.get(job.group.name, 0) + 1
    groups = ", ".join(f"{g} {n}" for g, n in by_group.items())
    typer.echo(f"plan:    {plan.name}: {len(jobs)} runs ({groups})")
    if dry_run:
        return

    settings = Settings()
    recording: Recording | None = None
    target = endpoint_from(record)
    if target.note:
        typer.echo(f"record:  {target.note}")
    if target.url:
        try:
            recording = start_recording(target.url, engine="langgraph")
        except RecordingUnavailable as error:
            typer.echo(f"error: {error}")
            raise typer.Exit(2) from error
        typer.echo(f"record:  traces to {shown(recording.url)}")

    def triage(job: Job, attempt: int) -> Outcome:
        models = resolve_models(
            job.group.model, job.group.comment_model, cheap=False, settings=settings
        )
        injector = FailureInjector(list(job.group.inject), rate=job.group.rate, seed=job.seed)
        attributes = {
            "triager.repo": data.repo.repo,
            "triager.case": job.case.id,
            "triager.engine": "langgraph",
            "triager.batch.group": job.group.name,
            "triager.batch.repeat": str(job.repeat),
            "triager.batch.attempt": str(attempt),  # a retried run leaves one trace per attempt
            "triager.inject": ",".join(job.group.inject),
        }
        root = (
            recording.triage_span(f"triage {data.repo.repo}#{job.case.id}", attributes)
            if recording
            else nullcontext()
        )
        trace_id = None
        try:
            with root as span:
                if recording is not None and span is not None:
                    trace_id = format(span.get_span_context().trace_id, "032x")
                state = run_langgraph(job.case, data, models, injector, yes=True, runs_dir=runs_dir)
                grades = finish_root(recording, span, state, job.case, models.triage_spec)
        except Exception as error:  # the root span has recorded it; keep its trace id
            raise RunError(error, trace_id) from error
        return Outcome(
            status=str(state.get("status")),
            grade=grades["triager.grade"],
            cost_usd=usage_cost(state.get("usage", [])),
            trace_id=trace_id,
            injected=tuple(injector.injected),
        )

    def progress(job: Job, outcome: Outcome) -> None:
        cost = "" if outcome.cost_usd is None else f" ${outcome.cost_usd:.4f}"
        result = outcome.grade or f"error: {outcome.error}"
        typer.echo(f"  {job.key:<28} {result}{cost}")

    try:
        summary = run_batch(
            jobs, triage, manifest=target_manifest, budget_usd=cap, on_done=progress
        )
    finally:
        if recording and (problem := recording.stop()):
            typer.echo(f"record:  {problem}")
    grades = ", ".join(f"{n} {g}" for g, n in sorted(summary.grades.items())) or "none"
    typer.echo(
        f"done:    {summary.done} runs ({grades}), {summary.errors} errors, "
        f"{summary.skipped} already in {target_manifest.as_posix()}; "
        f"spent ${summary.cost_usd:.2f}" + (f" of ${cap:.2f}" if cap is not None else "")
    )
    if summary.stopped == "budget":
        typer.echo("stopped: the spend cap was reached; raise --budget and run again to go on")
    why = {
        "billing": "the account is out of credit; top up",
        "daily quota": "its daily quota is spent; wait a day",
        "credentials": "its API key isn't set",
    }
    for provider, reason in summary.provider_stops.items():
        typer.echo(f"stopped: {provider}: {why.get(reason, reason)}, then run again to go on")
    if summary.not_run:
        typer.echo(f"not run: {summary.not_run} runs on stopped providers")
    if summary.errors or summary.provider_stops:
        raise typer.Exit(1)
