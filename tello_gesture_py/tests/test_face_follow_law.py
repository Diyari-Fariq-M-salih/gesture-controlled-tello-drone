"""No-camera checks for the face-follow forward/back law.

The distance law must command the same speed for the same standoff error on
either side of the target; the area law (as flown for arXiv v1) must stay
reproducible, asymmetry included.

    python -m tello_gesture_py.tests.test_face_follow_law
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tello_gesture_py.src.face_follow import FaceFollowConfig, FaceFollower  # noqa: E402

W, H = 960, 720
FRAME = np.zeros((H, W, 3), np.uint8)


def fb_at(distance_m, law):
    """Forward/back command with the face centred at `distance_m`."""
    cfg = FaceFollowConfig(fb_law=law, detect_every_n=10 ** 6)
    ff = FaceFollower(cfg)
    area = (cfg.k_dist_m / distance_m) ** 2
    side = int((area * W * H) ** 0.5)
    ff._last_bbox = (W // 2 - side // 2, H // 2 - side // 2, side, side)
    ff._last_area_frac = area
    ff._last_face_time = time.time()
    ff._last_control_ts = 0.0
    ff._frame_count = 1
    cmd, _ = ff.update(FRAME)
    return cmd.fb


def target():
    c = FaceFollowConfig()
    return c.k_dist_m / c.target_area_frac ** 0.5


def test_distance_law_is_symmetric():
    t = target()
    for e in (0.2, 0.4, 0.6):
        far, near = fb_at(t + e, "distance"), fb_at(t - e, "distance")
        assert far > 0 > near, (e, far, near)
        assert abs(far + near) <= 1, (e, far, near)


def test_distance_law_reaches_full_speed_forward():
    assert fb_at(target() + 2.0, "distance") == FaceFollowConfig().max_fb


def test_deadband_at_target():
    assert fb_at(target(), "distance") == 0
    assert fb_at(target(), "area") == 0


def test_area_law_unchanged():
    # arXiv v1 behaviour: slow approach (never above ~22), hard back-off.
    assert fb_at(3.0, "area") <= 22
    assert fb_at(0.5, "area") == -FaceFollowConfig().max_fb


def test_default_is_distance():
    assert FaceFollowConfig().fb_law == "distance"


if __name__ == "__main__":
    print(f"target {target():.2f} m")
    print(" distance   area law   distance law")
    for d in (0.5, 0.7, 0.94, 1.2, 1.5, 2.0, 3.0):
        print(f"  {d:4.2f} m   {fb_at(d, 'area'):+4d}       {fb_at(d, 'distance'):+4d}")
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
