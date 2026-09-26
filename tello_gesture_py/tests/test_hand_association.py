"""No-camera checks for hand-face association.

Every check from I. Chaabeni's test_hand_association_unit.py (Ilyes-branch) is
kept, adapted only to this tree's HandDetection (a single (21,3) `landmarks`
plus a `hands` list). Added: the tracker, the unmirrored Tello default, and the
single-hand path the paper's flights used staying unchanged.

    python -m pytest tello_gesture_py/tests/test_hand_association.py -q
    python -m tello_gesture_py.tests.test_hand_association          # no pytest
"""

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tello_gesture_py.src.config import ControllerConfig  # noqa: E402
from tello_gesture_py.src.hand_association import (  # noqa: E402
    AssociationTracker, HandAssociator)
from tello_gesture_py.src.hand_gesture import HandDetection  # noqa: E402

IMG_SHAPE = (720, 960, 3)          # H, W, C
AUTH_BBOX = (440, 100, 80, 100)    # px: origin_x, origin_y, width, height
FRAME = np.zeros(IMG_SHAPE, np.uint8)


def _lm(x, y, z=0.0, visibility=1.0):
    return types.SimpleNamespace(x=float(x), y=float(y), z=float(z),
                                 visibility=float(visibility))


def make_pose(left_hand=(0.30, 0.55), right_hand=(0.70, 0.55)):
    """A 33-landmark pose with the face inside AUTH_BBOX and shoulders 0.2 apart.

    Coordinates are image-space normalized, so `left_hand` is the pose's
    `left_*` group (indices 15/17/19/21) regardless of anatomy.
    """
    p = [_lm(0.5, 0.2) for _ in range(33)]
    h, w = IMG_SHAPE[:2]
    cx = (AUTH_BBOX[0] + AUTH_BBOX[2] / 2) / w
    cy = (AUTH_BBOX[1] + AUTH_BBOX[3] / 2) / h
    for i in (0, 1, 2, 3, 4, 5, 6, 9, 10):
        p[i] = _lm(cx, cy)
    p[11] = _lm(0.6, 0.35)   # left shoulder  -> shoulder width 0.2
    p[12] = _lm(0.4, 0.35)   # right shoulder
    for i in (15, 17, 19, 21):
        p[i] = _lm(*left_hand)
    for i in (16, 18, 20, 22):
        p[i] = _lm(*right_hand)
    return p


def make_hand(cx, cy, size=0.06, z_offset=0.0):
    """A (21,3) hand clustered around (cx, cy). `z_offset` fakes the Pose/Hands
    z-axis mismatch; a correct metric must be indifferent to it."""
    lm = np.zeros((21, 3), dtype=np.float32)
    rng = np.random.default_rng(0)
    for i in range(21):
        lm[i] = (cx + (rng.random() - 0.5) * size,
                 cy + (rng.random() - 0.5) * size,
                 z_offset)
    lm[0] = (cx, cy, z_offset)
    for i in (4, 8, 20):
        lm[i] = (cx, cy, z_offset)
    return lm


def hd(hands, labels=None, scores=None):
    """A multi-hand HandDetection in this tree's layout."""
    if not hands:
        return HandDetection(False, None)
    return HandDetection(True, hands[0], list(hands),
                         list(labels) if labels else [None] * len(hands),
                         list(scores) if scores else [None] * len(hands))


class FakeAssociator(HandAssociator):
    """HandAssociator with pose inference stubbed out (no model load)."""

    def __init__(self, poses, **kw):
        self._poses = [poses] if poses and not isinstance(poses[0], list) else (poses or [])
        self.detect_calls = 0
        self._configure(kw.get('target_side', 'right'), kw.get('mirrored', False),
                        kw.get('dist_thresh', 0.5), kw.get('side_margin', 0.85),
                        kw.get('face_frac_thresh', 0.6), kw.get('min_iou', 0.15),
                        kw.get('handedness_conf', 0.9),
                        kw.get('require_arm_visible', True), kw.get('arm_vis_thresh', 0.5))

    def detect(self, bgr):
        self.detect_calls += 1
        return types.SimpleNamespace(pose_landmarks=list(self._poses))


POSE = make_pose(left_hand=(0.30, 0.55), right_hand=(0.70, 0.55))
RIGHT = make_hand(0.70, 0.55, z_offset=-0.8)   # at the pose's right arm
LEFT = make_hand(0.30, 0.55, z_offset=0.9)     # at the pose's left arm
# On an UNMIRRORED frame MediaPipe Hands labels the arm Pose calls 'right' as
# 'Left': the two conventions are opposite. See _resolve_sides().
BOTH = hd([LEFT, RIGHT], ['Right', 'Left'], [0.99, 0.99])


# --- Ilyes's checks -----------------------------------------------------------

def test_metric_separates_the_two_arms():
    a = FakeAssociator(POSE)
    scale = a._pose_scale(POSE, AUTH_BBOX, IMG_SHAPE)
    d_rr = a.hand_against_skeleton(RIGHT, POSE, 'right', scale)
    d_lr = a.hand_against_skeleton(LEFT, POSE, 'right', scale)
    assert abs(scale - 0.2) < 1e-6, f"scale={scale}"
    assert d_rr < 0.1, f"right hand to right arm d={d_rr:.3f}"
    assert d_lr > 1.0, f"left hand to right arm d={d_lr:.3f}"
    assert d_lr > 10 * max(d_rr, 1e-6), "arms not separated by >10x"


def test_picks_the_target_arm_hand_in_any_order():
    a = FakeAssociator(POSE)
    got = a.associate_hand(FRAME, AUTH_BBOX, [BOTH])
    assert got is not None and len(got.hands) == 1, "should pick exactly one hand"
    assert np.allclose(got.landmarks, RIGHT), "picked the wrong hand"
    swapped = hd([RIGHT, LEFT], ['Left', 'Right'], [0.99, 0.99])
    got2 = a.associate_hand(FRAME, AUTH_BBOX, [swapped])
    assert got2 is not None and np.allclose(got2.landmarks, RIGHT), "order changed the pick"


def test_ambiguous_hand_is_rejected_not_guessed():
    a = FakeAssociator(make_pose(left_hand=(0.50, 0.55), right_hand=(0.52, 0.55)))
    assert a.associate_hand(FRAME, AUTH_BBOX, [hd([make_hand(0.51, 0.55)])]) is None


def test_hand_belonging_to_nobody_is_rejected():
    a = FakeAssociator(POSE)
    assert a.associate_hand(FRAME, AUTH_BBOX, [hd([make_hand(0.05, 0.95)])]) is None


def test_mirroring_flips_the_consulted_arm():
    a = FakeAssociator(POSE, mirrored=False)
    a_mir = FakeAssociator(POSE, mirrored=True)
    got = a_mir.associate_hand(FRAME, AUTH_BBOX, [hd([LEFT, RIGHT])])
    assert got is not None and np.allclose(got.landmarks, LEFT)
    assert a._resolve_sides() != a_mir._resolve_sides()


def test_confident_wrong_handedness_vetoes_unconfident_does_not():
    a = FakeAssociator(POSE)
    assert a.associate_hand(FRAME, AUTH_BBOX, [hd([RIGHT], ['Right'], [0.99])]) is None
    assert a.associate_hand(FRAME, AUTH_BBOX, [hd([RIGHT], ['Right'], [0.55])]) is not None


def test_face_gate_tolerates_tight_box_rejects_wrong_person():
    tight = make_pose()
    for i in (9, 10):
        tight[i] = _lm(0.5, 0.35)
    a = FakeAssociator(tight)
    idx, hits = a.find_skeleton(a.detect(FRAME), AUTH_BBOX, IMG_SHAPE)
    assert idx == 0, f"hits={hits}/8"
    away = make_pose()
    for i in (0, 1, 2, 3, 4, 5, 6, 9, 10):
        away[i] = _lm(0.05, 0.05)
    b = FakeAssociator(away)
    idx2, _ = b.find_skeleton(b.detect(FRAME), AUTH_BBOX, IMG_SHAPE)
    assert idx2 is None


def test_iou_tracks_small_moves_and_drops_jumps():
    a = FakeAssociator(POSE)
    prev = hd([RIGHT])
    assert a.update_hand_from_previous(prev, [hd([RIGHT + np.array([0.005, 0, 0], np.float32)])])
    assert a.update_hand_from_previous(prev, [hd([make_hand(0.2, 0.2)])]) is None


def test_repeated_tracking_does_not_corrupt_state():
    a = FakeAssociator(POSE)
    state = a.associate_hand(FRAME, AUTH_BBOX, [BOTH])
    for _ in range(5):
        _ = state.landmarks
        nxt = a.update_hand_from_previous(state, [BOTH])
        if nxt is None:
            break
        state = nxt
    assert np.allclose(state.landmarks, RIGHT)


def test_degenerate_inputs_return_none():
    a = FakeAssociator(POSE)
    assert FakeAssociator(None).associate_hand(FRAME, AUTH_BBOX, [BOTH]) is None
    assert a.associate_hand(FRAME, None, [BOTH]) is None
    assert a.associate_hand(FRAME, AUTH_BBOX, []) is None
    assert a.associate_hand(FRAME, AUTH_BBOX, [HandDetection(False, None)]) is None
    assert a.update_hand_from_previous(None, [BOTH]) is None
    flat = make_hand(0.7, 0.55) * 0
    assert a.update_hand_from_previous(hd([flat]), [hd([flat])]) is None


# --- added with the port ------------------------------------------------------

def test_bystander_skeleton_does_not_hijack_the_operator():
    # Two skeletons, the bystander listed first. Only the one whose face sits in
    # AUTH_BBOX may be used, so the bystander's hand at its own right arm loses.
    bystander = make_pose(left_hand=(0.05, 0.60), right_hand=(0.15, 0.60))
    for i in (0, 1, 2, 3, 4, 5, 6, 9, 10):
        bystander[i] = _lm(0.10, 0.20)
    a = FakeAssociator([bystander, POSE])
    their_hand = make_hand(0.15, 0.60)
    got = a.associate_hand(FRAME, AUTH_BBOX, [hd([their_hand, RIGHT])])
    assert got is not None and np.allclose(got.landmarks, RIGHT)
    assert a.associate_hand(FRAME, AUTH_BBOX, [hd([their_hand])]) is None


def test_deployed_default_is_unmirrored_and_off():
    cfg = ControllerConfig()
    assert cfg.hand_association is False, "association must be opt-in"
    assert cfg.assoc_mirrored is False, "the Tello feed is not mirrored"
    assert cfg.assoc_num_poses >= 2, "one pose lets a bystander crowd out the operator"


def test_tracker_runs_pose_then_iou_then_reanchors():
    a = FakeAssociator(POSE)
    t = AssociationTracker(a, every_n=3)
    modes = []
    for _ in range(5):
        t.step(FRAME, BOTH, AUTH_BBOX, fresh=True)
        modes.append(t.last.mode)
    assert modes == ["pose", "iou", "iou", "iou", "pose"], modes
    assert t.last.ok


def test_tracker_holds_on_throttled_frames_without_pose():
    a = FakeAssociator(POSE)
    t = AssociationTracker(a, every_n=20)
    first = t.step(FRAME, BOTH, AUTH_BBOX, fresh=True)
    calls = a.detect_calls
    held = t.step(FRAME, BOTH, AUTH_BBOX, fresh=False)
    assert held is first and t.last.mode == "held" and a.detect_calls == calls


def test_tracker_reacquires_immediately_when_iou_is_lost():
    a = FakeAssociator(POSE)
    t = AssociationTracker(a, every_n=20)
    t.step(FRAME, hd([make_hand(0.05, 0.05)]), AUTH_BBOX, fresh=True)   # nobody's hand
    assert not t.last.ok
    t.step(FRAME, BOTH, AUTH_BBOX, fresh=True)
    assert t.last.mode == "pose" and t.last.ok
    got = t.step(FRAME, hd([make_hand(0.2, 0.2), RIGHT + np.array([0.4, 0, 0], np.float32)]),
                 AUTH_BBOX, fresh=True)
    assert t.last.mode == "reacquire", t.last


def test_tracker_drops_state_when_face_or_hand_disappears():
    a = FakeAssociator(POSE)
    t = AssociationTracker(a, every_n=20)
    t.step(FRAME, BOTH, AUTH_BBOX, fresh=True)
    assert t.step(FRAME, BOTH, None, fresh=True) is None and t.last.reason == "no_face"
    t.step(FRAME, BOTH, AUTH_BBOX, fresh=True)
    assert t.last.mode == "pose", "a lost face must force a fresh pose association"
    assert t.step(FRAME, HandDetection(False, None), AUTH_BBOX, fresh=True) is None
    assert t.last.reason == "no_hand"


def test_arm_out_of_frame_is_rejected():
    # Pose returns every landmark even when the arm is out of shot; an elbow
    # below the frame means the arm linking hand to body was never seen.
    pose = make_pose()
    pose[14] = _lm(0.70, 1.15)                       # right elbow below the frame
    a = FakeAssociator(pose)
    assert a.associate_hand(FRAME, AUTH_BBOX, [BOTH]) is None
    assert a.last_info["reason"] == "arm_not_visible(elbow)", a.last_info


def test_low_visibility_elbow_is_rejected():
    pose = make_pose()
    pose[14] = _lm(0.70, 0.45, visibility=0.2)       # in frame but Pose is unsure
    a = FakeAssociator(pose)
    assert a.associate_hand(FRAME, AUTH_BBOX, [BOTH]) is None


def test_only_the_commanding_arm_must_be_visible():
    pose = make_pose()
    pose[13] = _lm(0.30, 1.20)                       # the OTHER arm's elbow off-frame
    a = FakeAssociator(pose)
    got = a.associate_hand(FRAME, AUTH_BBOX, [BOTH])
    assert got is not None and np.allclose(got.landmarks, RIGHT)


def test_arm_check_can_be_disabled_for_the_original_behaviour():
    pose = make_pose()
    pose[14] = _lm(0.70, 1.15)
    a = FakeAssociator(pose, require_arm_visible=False)
    got = a.associate_hand(FRAME, AUTH_BBOX, [BOTH])
    assert got is not None and np.allclose(got.landmarks, RIGHT)


def test_live_pose_mode_runs_pose_on_every_fresh_frame():
    # webcam_demo's 'k' key sets every_n = 0 for debugging.
    a = FakeAssociator(POSE)
    t = AssociationTracker(a, every_n=20)
    t.every_n = 0
    modes = []
    for _ in range(4):
        t.step(FRAME, BOTH, AUTH_BBOX, fresh=True)
        modes.append(t.last.mode)
    assert modes == ["pose"] * 4, modes


def test_live_pose_can_be_set_at_construction():
    # --live-pose passes every_n=0; it must not be clamped back to 1.
    a = FakeAssociator(POSE)
    t = AssociationTracker(a, every_n=0)
    for _ in range(3):
        t.step(FRAME, BOTH, AUTH_BBOX, fresh=True)
        assert t.last.mode == "pose"


def test_overlay_draws_with_and_without_association():
    import time as _t
    from tello_gesture_py.src.association_overlay import draw_debug
    a = FakeAssociator(POSE)
    a.last_pose = types.SimpleNamespace(pose_landmarks=[POSE])
    a.last_skeleton, a.last_pose_t = 0, _t.time()
    frame = np.zeros(IMG_SHAPE, np.uint8)
    draw_debug(frame, BOTH, hd([RIGHT]), a, AUTH_BBOX, True)
    green = (frame[..., 1] > 200) & (frame[..., 0] < 50) & (frame[..., 2] < 50)
    red = (frame[..., 2] > 200) & (frame[..., 1] < 50)
    assert green.any(), "accepted hand / face box not drawn in green"
    assert red.any(), "rejected hand not drawn in red"
    plain = np.zeros(IMG_SHAPE, np.uint8)
    draw_debug(plain, BOTH, None, None, AUTH_BBOX, False)   # association off
    assert plain.any(), "face box and hands should still be drawn"


def test_arm_check_is_on_by_default():
    assert ControllerConfig().assoc_require_arm is True


def test_single_hand_detection_keeps_its_deployed_shape():
    # What the paper's flights consumed: `landmarks` is one (21,3) array.
    d = HandDetection(True, RIGHT, [RIGHT], ['Left'], [0.9])
    assert d.landmarks.shape == (21, 3)
    assert HandDetection(False, None).hands == []


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
