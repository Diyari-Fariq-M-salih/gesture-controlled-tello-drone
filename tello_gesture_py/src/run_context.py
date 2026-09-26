"""Per-run output directory + provenance manifest.

Every launch of the controller gets its own directory under outputs/runs/, so
flight sessions never overwrite each other, and each one carries a manifest
recording exactly what code and configuration produced it.

Layout:
    outputs/runs/<YYYYmmdd-HHMMSS>_<run_id>/
        manifest.json     config snapshot, git SHA, library versions, host
        telemetry.csv     drone state at cfg.log_hz
        decisions.csv     mode/command/reason per decision
        perf.csv          per-frame stage latencies
        perf_events.csv   rc_send / llm_explanation events
        scenarios.csv     manual scenario trial outcomes
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _lib_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {"python": sys.version.split()[0]}
    for mod in ("cv2", "mediapipe", "numpy", "sklearn", "onnxruntime", "joblib"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return versions


class RunContext:
    """Owns one run's output directory and its manifest."""

    def __init__(self, run_id: Optional[str] = None, note: str = ""):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_id = run_id or "run"
        self.name = f"{stamp}_{self.run_id}"
        self.dir = RUNS_DIR / self.name
        self.dir.mkdir(parents=True, exist_ok=True)

        self.note = note
        self.started_at = time.time()
        self._manifest: Dict[str, Any] = {
            "run_id": self.run_id,
            "name": self.name,
            "note": note,
            "started_at": self.started_at,
            "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "libraries": _lib_versions(),
            "host": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "machine": platform.machine(),
            },
            "argv": sys.argv,
        }

    def path(self, filename: str) -> str:
        return str(self.dir / filename)

    def record(self, key: str, value: Any) -> None:
        """Add a section to the manifest (config, ablations, model paths...)."""
        if is_dataclass(value) and not isinstance(value, type):
            value = asdict(value)
        self._manifest[key] = value

    def write(self) -> None:
        self._manifest["ended_at"] = time.time()
        self._manifest["duration_s"] = self._manifest["ended_at"] - self.started_at
        (self.dir / "manifest.json").write_text(
            json.dumps(self._manifest, indent=2, default=str), encoding="utf-8"
        )
