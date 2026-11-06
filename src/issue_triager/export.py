"""Export any GitHub repo's issues and taxonomy to a dataset folder.

Read-only: it runs `gh` read commands and never writes to GitHub.
Hand-written files in the folder (synthetic.yaml, rules.yaml) are left alone.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from issue_triager.dataset import (
    GroundTruth,
    IssueExport,
    Label,
    Milestone,
    RepoInfo,
    TriageCase,
    dump_export,
    dump_repo,
)

ISSUE_FIELDS = "number,title,body,labels,milestone,state,stateReason,comments"
DUPLICATE_OF = re.compile(r"^\s*duplicate of #(\d+)", re.IGNORECASE | re.MULTILINE)
REGRESSION_OF = re.compile(r"^\s*regression of #(\d+)", re.IGNORECASE | re.MULTILINE)


def gh(*args: str) -> Any:
    out = subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout
    return json.loads(out)


def fetch_repo(repo: str) -> RepoInfo:
    info = gh("repo", "view", repo, "--json", "description")
    labels = gh("label", "list", "--repo", repo, "--limit", "500", "--json", "name,description")
    milestones = gh("api", f"repos/{repo}/milestones?state=all&per_page=100")
    milestones.sort(key=lambda m: (m.get("due_on") or "9999", m["number"]))
    return RepoInfo(
        repo=repo,
        description=info.get("description") or "",
        labels=sorted(
            (Label(name=lb["name"], description=lb.get("description") or "") for lb in labels),
            key=lambda lb: lb.name.lower(),
        ),
        milestones=[
            Milestone(
                title=m["title"],
                description=m.get("description") or "",
                due_on=(m.get("due_on") or "")[:10] or None,
                state=m["state"],
            )
            for m in milestones
        ],
    )


def fetch_issues(repo: str, limit: int = 1000) -> list[dict[str, Any]]:
    return gh(
        "issue", "list", "--repo", repo, "--state", "all", "--limit", str(limit),
        "--json", ISSUE_FIELDS,
    )  # fmt: skip


def duplicate_of(issue: dict[str, Any]) -> str | None:
    """GitHub's convention: a comment that starts "Duplicate of #N"."""
    for comment in issue.get("comments", []):
        if match := DUPLICATE_OF.search(comment["body"]):
            return match.group(1)
    return None


def regression_of(issue: dict[str, Any]) -> str | None:
    """Our convention: a line in the body or a comment that starts "Regression of #N"."""
    for text in (issue["body"] or "", *(c["body"] for c in issue.get("comments", []))):
        if match := REGRESSION_OF.search(text):
            return match.group(1)
    return None


def to_case(issue: dict[str, Any]) -> TriageCase:
    dupe, regression = duplicate_of(issue), regression_of(issue)
    return TriageCase(
        id=str(issue["number"]),
        title=issue["title"],
        body=(issue["body"] or "").replace("\r\n", "\n").strip(),
        truth=GroundTruth(
            labels=frozenset(label["name"] for label in issue["labels"]),
            milestone=(issue.get("milestone") or {}).get("title"),
            duplicate_of=dupe,
            regression_of=None if dupe else regression,
        ),
        source="real",
        state="closed" if issue["state"] == "CLOSED" else "open",
    )


def build_export(issues: list[dict[str, Any]], repo: str) -> IssueExport:
    cases = sorted((to_case(i) for i in issues), key=lambda c: int(c.id))
    return IssueExport(repo=repo, issues=cases)


def export(repo: str, out_dir: Path, *, limit: int = 1000) -> IssueExport:
    """Write repo.json and issues.json into `out_dir`."""
    info = fetch_repo(repo)
    issues = build_export(fetch_issues(repo, limit), repo)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "repo.json").write_text(dump_repo(info), encoding="utf-8", newline="\n")
    (out_dir / "issues.json").write_text(dump_export(issues), encoding="utf-8", newline="\n")
    return issues
