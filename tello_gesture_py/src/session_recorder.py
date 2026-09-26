"""Record the session to video inside the run directory.

The Tello stream arrives at a variable 4-24 fps, so frames are not written as
they arrive: a background thread writes the latest frame at a fixed rate against
the wall clock. Playback speed then matches real time, and frame i of the video
is time start + i / fps. Each written frame is stamped with that time and the
source frame's sequence number, so any moment in the video maps onto a row of
perf.csv (which logs seq) without guessing.

Two streams, both optional:
  session.mp4  what the operator saw: the frame with HUD and overlays
  raw.mp4      the camera frame before anything is drawn on it, for re-analysis

The videos show faces. outputs/runs/ is committed, so *.mp4 there is git-ignored.
"""
import threading
import time
from pathlib import Path
from typing import Optional

import cv2


class _Stream:
    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = fps
        self.writer = None
        self.latest = None      # (frame, seq)
        self.written = 0

    def open(self, shape):
        h, w = shape[:2]
        for fourcc, suffix in (("mp4v", ".mp4"), ("MJPG", ".avi")):
            path = self.path.with_suffix(suffix)
            wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), self.fps, (w, h))
            if wr.isOpened():
                self.writer, self.path = wr, path
                return True
            wr.release()
        return False


class SessionRecorder:
    def __init__(self, run_dir, fps: float = 30.0, record_raw: bool = False,
                 stamp: bool = True):
        run_dir = Path(run_dir)
        self.fps = float(fps)
        self.stamp = stamp
        self.streams = {"session": _Stream(run_dir / "session.mp4", self.fps)}
        if record_raw:
            self.streams["raw"] = _Stream(run_dir / "raw.mp4", self.fps)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.t0: Optional[float] = None
        self.write_ms_total = 0.0

    def submit(self, frame, seq=None, raw=None) -> None:
        """Hand over the newest frame (and optionally its raw copy). Cheap: the
        writer thread copies nothing and encodes on its own time."""
        with self._lock:
            if frame is not None:
                self.streams["session"].latest = (frame, seq)
            if raw is not None and "raw" in self.streams:
                self.streams["raw"].latest = (raw, seq)
        if self._thread is None and frame is not None:
            self.t0 = time.time()
            self._thread = threading.Thread(target=self._run, name="session-recorder", daemon=True)
            self._thread.start()

    def _write(self, s: _Stream, frame, seq, i: int) -> None:
        if self.stamp:
            frame = frame.copy()
            h = frame.shape[0]
            txt = f"t+{i / self.fps:8.2f}s  seq {seq if seq is not None else '-'}"
            cv2.putText(frame, txt, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        s.writer.write(frame)
        s.written += 1

    def _run(self) -> None:
        period = 1.0 / self.fps
        i = 0                      # index of the next video frame; it sits at t0 + i / fps
        self.skipped = 0
        while not self._stop.is_set():
            wait = self.t0 + i * period - time.time()
            if wait > 0:
                self._stop.wait(wait)
                if self._stop.is_set():
                    break
            with self._lock:
                items = [(s, s.latest) for s in self.streams.values() if s.latest is not None]
            # Slots due by now. Normally 1; more if encoding fell behind, in which
            # case the latest frame fills them, so frame i stays at time i / fps.
            due = int((time.time() - self.t0) / period) - i + 1
            fill = min(max(due, 1), int(self.fps))
            t_write = time.perf_counter()
            for k in range(fill):
                for s, (frame, seq) in items:
                    if s.writer is None and not s.open(frame.shape):
                        continue
                    self._write(s, frame, seq, i + k)
            self.write_ms_total += (time.perf_counter() - t_write) * 1000.0
            i += fill
            if due > fill:
                # More than a second behind: those slots are lost, and after this
                # point video time runs ahead of wall time by skipped / fps.
                self.skipped += due - fill
                i += due - fill

    def close(self) -> dict:
        """Stop, flush, and describe what was written (for the run manifest)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        out = {"fps": self.fps, "start_time": self.t0, "files": {}}
        for name, s in self.streams.items():
            if s.writer is not None:
                s.writer.release()
                out["files"][name] = {"path": s.path.name, "frames": s.written,
                                      "seconds": round(s.written / self.fps, 2)}
        frames = sum(s.written for s in self.streams.values())
        out["mean_write_ms_per_frame"] = round(self.write_ms_total / max(frames, 1), 2)
        out["slots_skipped"] = getattr(self, "skipped", 0)
        return out
