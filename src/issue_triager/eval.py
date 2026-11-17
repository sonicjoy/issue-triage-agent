"""The triager's eval suite for Agent Flight Recorder's harness (`afr eval`, AFR #15 and #16).

- `triage` is the adapter `afr eval run` calls once per case: it triages one dataset case on
  the model the case names and returns the four graded fields, normalised as the triager's
  own grader compares them, plus an ungraded `meta`.
- `build_suite` writes the suite's case files from the dataset and `evals/selection.yaml`
  (`triager suite`), so the dataset stays the only copy of every truth.
- `dataset_hash` is in every case's input: a hash of everything the agent can see besides
  the issue itself. A change to it makes every saved output stale.

Run from the repo root: `uv run afr eval run --suite triager`. See the README.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Self, cast

import yaml
from flight_recorder.evals import agent
from flight_recorder.evals.cases import dump
from flight_recorder.evals.checks import canonical
from flight_recorder.llm import chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, model_validator

from issue_triager.batch import classify
from issue_triager.dataset import DEFAULT_DATASET, TRUTH_FIELDS, Dataset, TriageCase, load_dataset
from issue_triager.grading import _normal  # pyright: ignore[reportPrivateUsage]  # one normaliser
from issue_triager.graph import Models, TriageState, build_graph
from issue_triager.tools import summarise

AGENT = "issue_triager.eval:triage"
SELECTION = Path("evals/selection.yaml")
SUITE_DIR = Path("evals/cases/triager")
SUITE_FILE = "suite.yaml"
WAITS = (15, 30)  # seconds before each retry of a transient provider error
SLEEP = time.sleep  # tests replace it


def dataset_hash(data: Dataset) -> str:
    """The first 12 hex digits of SHA-256 over what the agent can see besides the issue:
    every real issue as the search tools show it, the repo's labels and milestones, and
    the rules. Accepted answers and synthetic cases are left out: the agent can't see them.

    repo.json and rules.yaml are read from the files, not the loaded models, so no set's
    order can vary between processes. A missing rules.yaml (it's optional) hashes as null.
    """
    rules = data.path / "rules.yaml"
    seen = {
        "corpus": [summarise(c, full=True) for c in sorted(data.corpus.values(), key=_id)],
        "repo": json.loads((data.path / "repo.json").read_text(encoding="utf-8")),
        "rules": yaml.safe_load(rules.read_text(encoding="utf-8")) if rules.exists() else None,
    }
    return hashlib.sha256(canonical(seen).encode("utf-8")).hexdigest()[:12]


def _id(case: TriageCase) -> str:
    return case.id


def case_input(case: TriageCase, model: str, digest: str) -> dict[str, Any]:
    """A case's input: what the agent is given, the model to run, and the dataset hash."""
    return {**case.agent_input().model_dump(), "model": model, "dataset": digest}


# ---- the adapter -------------------------------------------------------------------


@agent
def triage(input: dict[str, Any]) -> dict[str, Any]:
    """Triage one case and return its graded fields plus an ungraded `meta`.

    Raises when the case was written for another dataset (hash, id, title or body differ),
    or when the model can't run (no key, a provider error that outlasts the retries). A
    decision that still breaks the rules after the agent's retry is returned, with
    `meta.status: "failed"`: it's a wrong answer, not an error.
    """
    data = load_dataset(DEFAULT_DATASET)
    case = _case_for(input, data)
    spec = cast(str, input["model"])
    state = _run(case, data, spec)
    decision = state.get("decision") or {}
    return {
        **{name: _normal(name, decision.get(name)) or None for name in TRUTH_FIELDS},
        "meta": {
            "model": spec,
            "status": state.get("status"),
            "feedback": state.get("feedback", []),
        },
    }


def _case_for(input: dict[str, Any], data: Dataset) -> TriageCase:
    for key in ("id", "title", "body", "model", "dataset"):
        if not isinstance(input.get(key), str):
            raise ValueError(f"the input's {key!r} must be text")
    if input["dataset"] != (digest := dataset_hash(data)):
        raise ValueError(
            f"the case was written for dataset {input['dataset']}, but this one is {digest}: "
            "regenerate the suite (`uv run triager suite`) and run it again"
        )
    case = data.by_id.get(input["id"])
    if case is None:
        raise ValueError(f"case {input['id']!r} isn't in the dataset")
    for key in ("title", "body"):
        if input[key] != getattr(case, key):
            raise ValueError(f"case {case.id}: the input's {key} differs from the dataset's")
    return case


def _run(case: TriageCase, data: Dataset, spec: str) -> dict[str, Any]:
    """`_run_once`, tried again after each wait in WAITS when the provider error is a
    transient one (`batch.classify`: a rate limit that isn't a daily quota, a 503 or 529).
    Any other error is raised at once. The provider SDKs already retry each call twice;
    this covers the longer outages the corpus saw."""
    for wait in WAITS:
        try:
            return _run_once(case, data, spec)
        except Exception as error:
            if classify(error)[0] != "retry":
                raise
            SLEEP(wait)
    return _run_once(case, data, spec)


def _run_once(case: TriageCase, data: Dataset, spec: str) -> dict[str, Any]:
    """One triage, as `triager run --yes` does it but without printing the review (which
    shows the truth). The dry-run file goes to a temporary folder, deleted afterwards."""
    triage_model = chat_model(spec)
    comment_model = chat_model(spec)
    models = Models(triage_model, comment_model, triage_spec=spec, comment_spec=spec)
    with tempfile.TemporaryDirectory(prefix="triager-eval-") as runs_dir:
        graph = build_graph(
            case, data, models, checkpointer=InMemorySaver(), runs_dir=Path(runs_dir)
        )
        config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
        state = graph.invoke(cast(TriageState, {"case_id": case.id}), config)
        if "__interrupt__" in state:
            state = graph.invoke(Command(resume=True), config)
    return dict(state)


# ---- the generator -----------------------------------------------------------------


class Selected(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", coerce_numbers_to_str=True)

    id: str
    """A dataset id. An unquoted 3 reads as "3"."""
    tags: list[str]
    """Free text for people: the failure modes and fields the case covers."""
    note: str
    """What the corpus shows, then why the case is in."""


class Selection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    """The model spec every case runs on (both the triage and comment calls)."""
    cases: list[Selected]

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [c.id for c in self.cases]
        if repeated := sorted({i for i in ids if ids.count(i) > 1}):
            raise ValueError(f"cases listed twice: {', '.join(repeated)}")
        return self


def load_selection(path: Path) -> Selection:
    """Raises OSError, or ValueError (pydantic's ValidationError is one) for a bad file."""
    return Selection.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def checks_for(case: TriageCase) -> list[dict[str, Any]]:
    """One check per field from the truth: `one_of` (the truth, then each other accepted
    answer once) when the field has accepted answers, `equals` otherwise. Nothing is null."""
    found: list[dict[str, Any]] = []
    for name in TRUTH_FIELDS:
        answers: list[Any] = []
        for answer in (getattr(case.truth, name), *getattr(case.truth.accept, name)):
            value = _normal(name, answer) or None
            if value not in answers:
                answers.append(value)
        if len(answers) > 1:
            found.append({"type": "one_of", "field": name, "values": answers})
        else:
            found.append({"type": "equals", "field": name, "value": answers[0]})
    return found


def build_suite(data: Dataset, selection: Selection) -> dict[str, str]:
    """The suite's files, name to text: `suite.yaml` and one `<id>.yaml` per selected case.
    Raises ValueError for an id that isn't in the dataset."""
    digest = dataset_hash(data)
    files = {SUITE_FILE: dump({"agent": AGENT, "model": selection.model})}
    for selected in selection.cases:
        case = data.by_id.get(selected.id)
        if case is None:
            raise ValueError(f"case {selected.id!r} in the selection isn't in the dataset")
        files[f"{case.id}.yaml"] = dump({
            "id": case.id,
            "tags": list(selected.tags),
            "note": selected.note,
            "input": case_input(case, selection.model, digest),
            "checks": checks_for(case),
        })  # fmt: skip
    return files


def write_suite(files: dict[str, str], folder: Path = SUITE_DIR) -> None:
    """Write the case files and delete case files no longer selected. outputs.jsonl and
    baseline.json, written by `afr eval`, are left alone."""
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("*.yaml"):
        if stale.name not in files:
            stale.unlink()
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8", newline="\n")
