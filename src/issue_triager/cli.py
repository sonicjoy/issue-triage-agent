"""Triage GitHub issues with a LangGraph agent (dry run: nothing is written to GitHub)."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Annotated, Any, cast

import typer
from flight_recorder.llm import MissingCredentialsError, ModelSpecError, chat_model
from flight_recorder.pricing import cost_usd
from flight_recorder.settings import Settings
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from issue_triager.dataset import (
    DATASETS_DIR,
    DEFAULT_DATASET,
    GroundTruth,
    TriageCase,
    dataset_dir_name,
    load_dataset,
)
from issue_triager.export import export as export_dataset
from issue_triager.graph import Models, TriageState, Usage, build_graph
from issue_triager.tools import FAILURE_MODES, FailureInjector, FailureMode

app = typer.Typer(add_completion=False, help=__doc__)


def resolve_models(
    model: str | None, comment_model: str | None, *, cheap: bool, settings: Settings
) -> Models:
    """--model sets both steps unless --comment-model is given; otherwise use settings."""
    triage_spec = model or (settings.llm_cheap_model if cheap else settings.llm_model)
    comment_spec = comment_model or model or settings.llm_cheap_model
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
    marks: list[str] = []
    for name in ("labels", "milestone", "duplicate_of", "regression_of"):
        got, want = decision.get(name), getattr(truth, name)
        if name == "labels":
            got, want = sorted(got or []), sorted(want)
        elif isinstance(got, str):
            got = got.lstrip("#")
        marks.append(f"{name} {'ok' if got == want else f'WRONG (want {want})'}")
    return "; ".join(marks)


def report_usage(usage: list[Usage]) -> str:
    total = 0.0
    priced = True
    for row in usage:
        cost = cost_usd(row["model"], row["input_tokens"], row["output_tokens"])
        if cost is None:
            priced = False
        else:
            total += cost
    tokens_in = sum(r["input_tokens"] for r in usage)
    tokens_out = sum(r["output_tokens"] for r in usage)
    cost_text = f"${total:.6f}" if priced else "unknown (unpriced model)"
    return f"tokens:  {tokens_in} in / {tokens_out} out   cost: {cost_text}"


def show_review(state: dict[str, Any], case: TriageCase) -> None:
    for step in state.get("steps", []):
        status = step.get("status", "?")
        typer.echo(f"  [{step['attempt']}] {step['tool']}({step['args']}) -> {status}")
    decision = state.get("decision") or {}
    typer.echo(f"decision: {decision}")
    typer.echo(f"vs truth: {grade(decision, case.truth)}")
    typer.echo(f"comment:\n{state.get('comment', '')}\n")


@app.command()
def run(
    issue: Annotated[str, typer.Option(help='Issue number ("7") or synthetic case id ("s53")')],
    dataset: Annotated[
        Path, typer.Option(help="Dataset folder from `triager export`")
    ] = DEFAULT_DATASET,
    model: Annotated[str | None, typer.Option(help="<provider>:<model> for both steps")] = None,
    comment_model: Annotated[str | None, typer.Option(help="Model for the comment")] = None,
    cheap: Annotated[bool, typer.Option(help="Triage with the cheap model")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Approve without asking")] = False,
    inject: Annotated[str, typer.Option(help=f"Tool failures: {','.join(FAILURE_MODES)}")] = "",
    rate: Annotated[float, typer.Option(min=0, max=1, help="Chance per tool call")] = 0.3,
    seed: Annotated[int | None, typer.Option(help="Seed for repeatable injection")] = None,
    runs_dir: Annotated[Path, typer.Option(help="Where 'apply' writes")] = Path("runs"),
) -> None:
    """Triage one issue from a dataset end to end."""
    if not (dataset / "issues.json").exists():
        typer.echo(f"no dataset at {dataset}; create one with `triager export OWNER/NAME`")
        raise typer.Exit(2)
    data = load_dataset(dataset)
    case = data.by_id.get(issue.lstrip("#"))
    if case is None:
        typer.echo(f"no issue {issue!r} in {data.repo.repo} ({len(data.real)} real issues)")
        raise typer.Exit(2)
    try:
        models = resolve_models(model, comment_model, cheap=cheap, settings=Settings())
    except (ModelSpecError, MissingCredentialsError) as error:
        typer.echo(str(error))
        raise typer.Exit(2) from error

    injector = FailureInjector(parse_inject(inject), rate=rate, seed=seed)
    graph = build_graph(
        case, data, models, checkpointer=InMemorySaver(), injector=injector,
        runs_dir=runs_dir,
    )  # fmt: skip
    config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
    typer.echo(f"triaging {data.repo.repo}#{case.id} ({case.source}): {case.title}")
    typer.echo(f"models:  triage {models.triage_spec}, comment {models.comment_spec}")

    started = time.perf_counter()
    try:
        state = graph.invoke(cast(TriageState, {"case_id": case.id}), config)
        if "__interrupt__" in state:
            show_review(dict(state), case)
            approved = yes or typer.confirm("Apply (dry run: writes a local file)?", default=False)
            state = graph.invoke(Command(resume=approved), config)
    except Exception as error:  # provider and network errors: one clear line
        first_line = str(error).splitlines()[0] if str(error) else ""
        typer.echo(f"error: {type(error).__name__}: {first_line}")
        raise typer.Exit(1) from error

    if state.get("status") == "failed":
        show_review(dict(state), case)
        typer.echo(
            f"failed validation after {state.get('attempts')} attempts: {state.get('feedback')}"
        )
    if injector.injected:
        typer.echo(f"injected: {', '.join(injector.injected)}")
    typer.echo(report_usage(state.get("usage", [])))
    typer.echo(f"status:  {state.get('status')}  {state.get('output_path', '')}".rstrip())
    typer.echo(f"latency: {time.perf_counter() - started:.1f}s")
    if state.get("status") == "failed":
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
