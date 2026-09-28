"""No-camera checks for follow-behind: the re-ID model, the control law, arbitration.

    python -m tello_gesture_py.tests.test_back_follow
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tello_gesture_py.src.back_follow import BackFollower, BackObs, BackReID  # noqa: E402
from tello_gesture_py.src.config import DEFAULT_REID_MODEL, BackFollowConfig  # noqa: E402
from tello_gesture_py.src.mode_manager import DeterministicModeManager  # noqa: E402

W = 960


def _follower():
    """The control half only: no pose or re-ID model needed."""
    f = object.__new__(BackFollower)
    f.cfg = BackFollowConfig()
    return f


def _obs(cx, dist):
    return BackObs(detected=True, bbox=(int(cx - 60), 100, int(cx + 60), 500), score=0.8, dist_m=dist)


def test_control_directions():
    f = _follower()
    d = f.cfg.follow_distance_m
    assert f.command(_obs(W / 2 + 200, d), W, None).yaw > 0      # operator right: turn right
    assert f.command(_obs(W / 2 - 200, d), W, None).yaw < 0
    assert f.command(_obs(W / 2, d + 1.5), W, None).fb > 0       # too far: forward
    assert f.command(_obs(W / 2, d - 1.0), W, None).fb < 0       # too close: back
    c = f.command(_obs(W / 2, d), W, None)
    assert (c.fb, c.yaw) == (0, 0)                                # on target: hold
    assert f.command(_obs(W / 2, d), W, 60).ud > 0                # below set height: climb
    assert f.command(_obs(W / 2, d), W, 300).ud < 0              # above set height: descend


def test_lost_operator_hovers():
    c = _follower().command(BackObs(detected=False), W, 100)
    assert (c.lr, c.fb, c.ud, c.yaw) == (0, 0, 0, 0) and c.active


def test_distance_estimate():
    f = _follower()
    # a 0.5 m torso 136 px long is 2.5 m away at the Tello focal length
    assert abs(f.distance_m(136.0, W) - 2.5) < 0.01


def test_setpoint_keys():
    f = _follower()
    d, h = f.cfg.follow_distance_m, f.cfg.follow_height_m
    f.adjust(ord("]"))
    f.adjust(ord("="))
    assert f.cfg.follow_distance_m == d + 0.25 and abs(f.cfg.follow_height_m - (h + 0.1)) < 1e-9
    assert f.adjust(ord("x")) is None


def test_arbitration():
    base = {"hand_detected": False, "face_detected": False, "time_since_any_seen_s": 0.0}
    m = DeterministicModeManager()
    assert m.tick(dict(base, back_detected=True, time_since_back_s=0.0))[0] == "follow_back"
    assert m.tick(dict(base, face_detected=True, back_detected=True))[0] == "face"
    m = DeterministicModeManager()
    assert m.tick(dict(base, hand_detected=True, back_detected=True))[0] == "gesture"
    # without the feature nothing reports a back: behaviour as before
    assert DeterministicModeManager().tick(dict(base))[0] == "hover"


def test_reid_model():
    if not os.path.exists(DEFAULT_REID_MODEL):
        print("    (skipped: re-ID model not built; see export_osnet.py)")
        return
    reid = BackReID(str(DEFAULT_REID_MODEL))
    rng = np.random.default_rng(0)
    person = np.zeros((256, 128, 3), np.uint8)
    person[:120] = (40, 40, 200)                      # red shirt
    person[120:] = (120, 60, 20)                      # blue trousers
    person = np.clip(person + rng.integers(0, 20, person.shape), 0, 255).astype(np.uint8)
    brighter = np.clip(person.astype(int) + 25, 0, 255).astype(np.uint8)
    other = rng.integers(0, 255, person.shape).astype(np.uint8)
    e = reid.embed([person, person, brighter, other])
    assert e.shape == (4, 512)
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)
    assert e[0] @ e[1] > 0.999                        # deterministic
    assert e[0] @ e[2] > e[0] @ e[3]                  # lighting change closer than noise


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
