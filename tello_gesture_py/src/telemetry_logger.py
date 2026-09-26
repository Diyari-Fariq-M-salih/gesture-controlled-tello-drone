import csv
import time
from typing import Dict, Iterable, List, Optional


from pathlib import Path

# src/ lives at <project-root>/tello_gesture_py/src, so parents[2] is the root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    """Drone state sampled at a fixed rate.

    The caller may invoke add() every control-loop iteration; samples are
    dropped to honour `hz` so the exported rate matches the documented one.
    Pass hz=0 to record every call.
    """

    def __init__(self, fields: Iterable[str], path: str, hz: float = 1.0):
        self.fields = list(fields)
        self.path = _resolve_output_csv_path(path)
        self.hz = float(hz)
        self.rows: List[Dict[str, object]] = []
        self._min_dt = (1.0 / self.hz) if self.hz > 0 else 0.0
        self._last_ts = 0.0

    def add(self, state: Dict[str, float]) -> bool:
        now = time.time()
        if self._min_dt and (now - self._last_ts) < self._min_dt:
            return False
        self._last_ts = now
        row: Dict[str, object] = {"t": now}
        for k in self.fields:
            row[k] = state.get(k, None)
        self.rows.append(row)
        return True

    def export(self):
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t"] + self.fields)
            w.writeheader()
            w.writerows(self.rows)


class DecisionLogger:
    """CSV logger for mode decisions, deterministic reasons and LLM explanations."""

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
            "mode",
            "command",
            "det_reason",
            "gesture",
            "gesture_conf",
            "face_raw",
            "face_auth",
            "hand_raw",
            "hand_auth",
            "battery",
            "altitude_cm",
            "llm_reason",
            "llm_latency_ms",
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


class ScenarioLogger:
    """Manual trial logger for scenario-level system tests.

    Operator presses a digit key to open a trial for a named scenario, then
    'y'/'n' to record pass/fail once the expected behavior is observed, or
    'x' to discard a trial that was mis-started or interrupted.

    Each row captures the mode the system was in when the trial opened, which
    lets you verify afterwards that a trial actually started from the intended
    precondition rather than trusting the operator's memory.
    """

    def __init__(self, path: str, run_id: str = ""):
        self.path = _resolve_output_csv_path(path)
        self.run_id = run_id
        self.rows: List[Dict[str, object]] = []
        self._open: Optional[Dict[str, object]] = None

    @property
    def open_trial(self) -> Optional[Dict[str, object]]:
        return self._open

    def start(self, scenario: str, mode_at_start: str = "", battery: Optional[float] = None):
        if self._open is not None:
            print(f"[Scenario] Discarding unresolved trial: {self._open['scenario']} #{self._open['trial']}")
        trial_id = sum(1 for r in self.rows if r["scenario"] == scenario) + 1
        self._open = {
            "run_id": self.run_id,
            "scenario": scenario,
            "trial": trial_id,
            "t_start": time.time(),
            "mode_at_start": mode_at_start,
            "battery_at_start": battery,
        }
        print(f"[Scenario] START '{scenario}' trial #{trial_id} (mode={mode_at_start}, bat={battery}) "
              f"-- y=pass  n=fail  x=discard")

    def discard(self):
        if self._open is None:
            print("[Scenario] No open trial to discard.")
            return
        print(f"[Scenario] DISCARDED '{self._open['scenario']}' trial #{self._open['trial']}")
        self._open = None

    def resolve(self, success: bool, mode_at_end: str = ""):
        if self._open is None:
            print("[Scenario] No open trial -- press a scenario key (1-8) first.")
            return
        row = dict(self._open)
        row["t_end"] = time.time()
        row["duration_s"] = row["t_end"] - row["t_start"]
        row["mode_at_end"] = mode_at_end
        row["success"] = bool(success)
        self.rows.append(row)
        n = sum(1 for r in self.rows if r["scenario"] == row["scenario"])
        ok = sum(1 for r in self.rows if r["scenario"] == row["scenario"] and r["success"])
        print(f"[Scenario] '{row['scenario']}' trial #{row['trial']}: "
              f"{'PASS' if success else 'FAIL'}   [{ok}/{n} this run]")
        self._open = None

    def tally(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for r in self.rows:
            s = str(r["scenario"])
            n = sum(1 for x in self.rows if x["scenario"] == s)
            ok = sum(1 for x in self.rows if x["scenario"] == s and x["success"])
            out[s] = f"{ok}/{n}"
        return out

    def export(self):
        fieldnames = [
            "run_id", "scenario", "trial", "t_start", "t_end", "duration_s",
            "mode_at_start", "mode_at_end", "battery_at_start", "success",
        ]
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.rows)
