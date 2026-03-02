import csv
import time
from typing import Dict, Iterable, List
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_output_csv_path(p: str) -> str:
    P = Path(p)
    if P.is_absolute():
        P.parent.mkdir(parents=True, exist_ok=True)
        return str(P)

    # if just a filename, put it under outputs/
    if P.parent == Path("."):
        out = PROJECT_ROOT / "outputs" / P.name
        out.parent.mkdir(parents=True, exist_ok=True)
        return str(out)

    # otherwise resolve relative to project root
    out = PROJECT_ROOT / P
    out.parent.mkdir(parents=True, exist_ok=True)
    return str(out)


class TelemetryLogger:
    def __init__(self, fields: Iterable[str], path: str):
        self.fields = list(fields)
        self.path = _resolve_output_csv_path(path)
        self.rows: List[Dict[str, object]] = []

    def add(self, state: Dict[str, float]):
        row = {"t": time.time()}
        for k in self.fields:
            row[k] = state.get(k, None)
        self.rows.append(row)

    def export(self):
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t"] + self.fields)
            w.writeheader()
            w.writerows(self.rows)


class DecisionLogger:
    """CSV logger for LLM decisions + reasons."""

    def __init__(self, path: str):
        self.path = _resolve_output_csv_path(path)
        self.rows: List[Dict[str, object]] = []

    def add(self, row: Dict[str, object]):
        if "t" not in row:
            row = dict(row)
            row["t"] = time.time()
        self.rows.append(row)

    def export(self):
        # Always write a file (even if empty), so user can find it.
        preferred = [
            "t",
            "mode_before",
            "mode_after",
            "gesture",
            "gesture_conf",
            "face_detected",
            "time_since_face_s",
            "hand_detected",
            "time_since_hand_s",
            "key",
            "battery",
            "altitude_cm",
            "reason",
        ]

        keys = set(preferred)
        for r in self.rows:
            keys.update(r.keys())

        fieldnames = [k for k in preferred if k in keys] + [
            k for k in sorted(keys) if k not in preferred
        ]

        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.rows)