"""No-camera checks for the session recorder.

    python -m pytest tello_gesture_py/tests/test_session_recorder.py -q
    python -m tello_gesture_py.tests.test_session_recorder          # no pytest
"""

import os
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tello_gesture_py.src.session_recorder import SessionRecorder  # noqa: E402

FPS = 20.0


def _feed(rec, seconds, src_fps, raw=False):
    """A jittery source slower than the recorder, as the Tello stream is."""
    rng = np.random.default_rng(0)
    t_end = time.time() + seconds
    seq = 0
    while time.time() < t_end:
        f = np.full((240, 320, 3), seq % 255, np.uint8)
        rec.submit(f, seq=seq, raw=f.copy() if raw else None)
        seq += 1
        time.sleep(max(0.0, rng.normal(1.0 / src_fps, 0.3 / src_fps)))
    return seq


def _frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def test_video_runs_in_real_time():
    with tempfile.TemporaryDirectory() as d:
        rec = SessionRecorder(d, fps=FPS)
        _feed(rec, 2.0, src_fps=8)
        info = rec.close()
        s = info["files"]["session"]
        assert abs(s["seconds"] - 2.0) < 0.3, s
        assert _frames(os.path.join(d, s["path"])) == s["frames"]
        assert "raw" not in info["files"]
        assert info["slots_skipped"] == 0


def test_raw_stream_matches_session():
    with tempfile.TemporaryDirectory() as d:
        rec = SessionRecorder(d, fps=FPS, record_raw=True)
        _feed(rec, 1.0, src_fps=10, raw=True)
        info = rec.close()
        assert set(info["files"]) == {"session", "raw"}
        a, b = info["files"]["session"]["frames"], info["files"]["raw"]["frames"]
        assert abs(a - b) <= 1, (a, b)


def test_no_frames_writes_nothing():
    with tempfile.TemporaryDirectory() as d:
        info = SessionRecorder(d, fps=FPS).close()
        assert info["files"] == {} and os.listdir(d) == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
