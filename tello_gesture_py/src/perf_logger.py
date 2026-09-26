"""Per-frame stage timing + event-rate logging for runtime performance tables.

Produces the "Runtime Performance" numbers: FPS, per-stage latency,
camera-to-command latency, RC command rate, LLM explanation latency.

Two measurement rules matter for the numbers to mean anything:

1. Stages that are throttled (hand detection every N frames, face every M)
   must only contribute a latency sample on frames where they actually ran.
   Averaging real detections together with skipped frames produces a figure
   that describes neither -- it is what makes an untimed skip frame look like
   a sub-millisecond face detector. `StageTimer.mark(stage, ran=...)` records a
   duration only when ran=True and always emits a `<stage>_ran` column, so
   downstream `dropna()` yields the conditional distribution.

2. A frame must be logged once. The controller loop can spin faster than the
   video decoder delivers, so the same decoded frame can be observed many
   times; logging each observation inflates the instantaneous frame rate.
   Callers pass the frame sequence number and PerfLogger drops repeats.
"""

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_output_csv_path(p: str) -> str:
    P = Path(p)
    if P.is_absolute():
        P.parent.mkdir(parents=True, exist_ok=True)
        return str(P)
    if P.parent == Path("."):
        out = PROJECT_ROOT / "outputs" / "metrics" / P.name
        out.parent.mkdir(parents=True, exist_ok=True)
        return str(out)
    out = PROJECT_ROOT / P
    out.parent.mkdir(parents=True, exist_ok=True)
    return str(out)


class StageTimer:
    """Call .mark(stage, ran=True/False) after each pipeline stage for a frame.

    Pass ran=False when the stage was skipped this frame (throttled, or its
    precondition was absent). The clock still advances -- so the skip cost is
    not billed to the next stage -- but no latency sample is recorded.
    """

    def __init__(self):
        self._t0 = time.perf_counter()
        self._last = self._t0
        self.durations: Dict[str, float] = {}
        self.ran: Dict[str, bool] = {}

    def mark(self, stage: str, ran: bool = True) -> Optional[float]:
        now = time.perf_counter()
        dt_ms = (now - self._last) * 1000.0
        self._last = now
        self.ran[stage] = bool(ran)
        if not ran:
            return None
        self.durations[stage] = dt_ms
        return dt_ms

    def record(self, stage: str, ms: float) -> None:
        """Record a nested sub-measurement timed by the caller.

        Use for a stage measured inside another stage (e.g. classifier
        inference inside command execution). Does not advance the clock, so
        the enclosing stage still accounts for the same interval.
        """
        self.durations[stage] = ms
        self.ran[stage] = True

    def elapsed_ms(self) -> float:
        """Total time since the timer was created, regardless of marks."""
        return (time.perf_counter() - self._t0) * 1000.0


class PerfLogger:
    """One row per *distinct* processed frame, plus named events."""

    def __init__(self, path: str):
        self.path = _resolve_output_csv_path(path)
        self.rows: List[Dict[str, object]] = []
        self._last_frame_ts: Optional[float] = None
        self._last_seq: Optional[int] = None
        self.duplicate_frames = 0

        events_path = str(Path(self.path).with_name(Path(self.path).stem + "_events.csv"))
        self.events_path = events_path
        self._event_rows: List[Dict[str, object]] = []

    def log_frame(
        self,
        timer: StageTimer,
        camera_to_command_ms: float,
        extra: Optional[Dict[str, object]] = None,
        seq: Optional[int] = None,
        frame_age_ms: Optional[float] = None,
    ) -> bool:
        """Record one frame. Returns False if it was a duplicate and was skipped."""
        if seq is not None and seq == self._last_seq:
            self.duplicate_frames += 1
            return False
        self._last_seq = seq

        now = time.time()
        dt_s = None
        fps_inst = None
        if self._last_frame_ts is not None:
            dt_s = now - self._last_frame_ts
            if dt_s > 0:
                fps_inst = 1.0 / dt_s
        self._last_frame_ts = now

        row: Dict[str, object] = {
            "t": now,
            "seq": seq,
            "dt_s": dt_s,
            "fps_inst": fps_inst,
            "frame_age_ms": frame_age_ms,
            "camera_to_command_ms": camera_to_command_ms,
        }
        row.update(timer.durations)
        row.update({f"{k}_ran": int(v) for k, v in timer.ran.items()})
        if extra:
            row.update(extra)
        self.rows.append(row)
        return True

    def log_event(self, name: str, latency_ms: Optional[float] = None):
        """Record an occurrence of a named event (e.g. 'rc_send', 'llm_explanation')."""
        self._event_rows.append({"t": time.time(), "event": name, "latency_ms": latency_ms})

    def export(self):
        if self.rows:
            leading = ["t", "seq", "dt_s", "fps_inst", "frame_age_ms", "camera_to_command_ms"]
            keys = set()
            for r in self.rows:
                keys.update(r.keys())
            fieldnames = leading + [k for k in sorted(keys) if k not in leading]
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(self.rows)

        with open(self.events_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t", "event", "latency_ms"])
            w.writeheader()
            w.writerows(self._event_rows)
