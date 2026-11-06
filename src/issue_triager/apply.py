"""The 'apply' step, shared by every version of the agent.

Dry run: it writes what would be applied to a local file. Nothing goes to GitHub.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_dry_run(runs_dir: Path, case_id: str, record: dict[str, Any]) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = runs_dir / f"triage-{case_id}-{stamp}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
