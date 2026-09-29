"""Hand-face association: which detected hand belongs to the authorized person.

The identity gate establishes that the enrolled operator is *present*; on its
own it does not bind the commanding hand to them, so a bystander's hand can
drive the aircraft while the operator stands in frame. This module closes that
gap through the body: a MediaPipe Pose skeleton is matched to the verified face
box, and a hand is accepted only if it sits at the end of that skeleton's
target arm.

HandAssociator is I. Chaabeni's design (Ilyes-branch, 2026-09-22/23); its
selection rules are unchanged here. Changes made in porting it to this tree:
  - num_poses defaults to 2: MediaPipe's default of 1 can lock onto a bystander,
    after which no skeleton matches the face and every gesture is dropped;
  - segmentation masks are no longer computed (they were never read);
  - the model is loaded from bytes, which avoids MediaPipe's path handling on
    Windows, and lives in models/mediapipe/;
  - the Tello feed is not mirrored, so the deployed default is mirrored=False;
  - the target arm's elbow and wrist must be in frame and visible: Pose
    invents an out-of-frame arm from the visible hand, which otherwise lets a
    hand with no arm in shot match its own guessed arm.

AssociationTracker holds the per-frame state (when to re-run pose, when to
carry the hand by its palm track, when to coast) so the flight controller and
the webcam harness run the same code.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .hand_gesture import HandDetection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POSE_MODEL = PROJECT_ROOT / "models" / "mediapipe" / "pose_landmarker_full.task"


PALM_IDX = (0, 5, 9, 13, 17)   # palm landmarks used for tracking: wrist + 4 MCPs


@dataclass
class _HandTrack:
    """The tracked hand's palm, in full-frame pixels."""
    anchor: np.ndarray   # mean of palm points, px
    bbox: np.ndarray     # palm bbox [x1, y1, x2, y2], px
    size: float          # palm size, px
    missed: int = 0      # consecutive frames with no valid candidate


def _candidates(det):
    """(landmarks, handedness label, handedness score) for every hand in `det`."""
    if det is None or not det.has_hand:
        return []
    if det.hands:
        n = len(det.hands)
        labels = det.handedness if len(det.handedness) == n else [None] * n
        scores = det.handedness_score if len(det.handedness_score) == n else [None] * n
        return list(zip(det.hands, labels, scores))
    if det.landmarks is not None:
        return [(det.landmarks, None, None)]
    return []


def _single(lm, label=None, score=None) -> HandDetection:
    return HandDetection(True, lm, [lm], [label], [score])


class HandAssociator:
    """Picks the single hand belonging to the authorized person's target arm.

    Two entry points, meant to alternate:
      - associate_hand(): full pose inference, anchors the hand to the authorized
        skeleton. Accurate but expensive, so it runs only every N frames. Follow a
        successful call with start_track() to seed the cheap tracker.
      - update_hand_from_previous(): cheap palm-track carry-over of the
        already-associated hand across the frames in between. Reports "coasting"
        when it sees no acceptable candidate but the track is still worth holding,
        and "lost" when the caller must fall back to associate_hand().
    """

    def __init__(self,
                 target_side: str = 'right',
                 mirrored: bool = False,
                 dist_thresh: float = 0.5,
                 side_margin: float = 0.85,
                 face_frac_thresh: float = 0.6,
                 handedness_conf: float = 0.9,
                 num_poses: int = 2,
                 model_path=None,
                 require_arm_visible: bool = True,
                 arm_vis_thresh: float = 0.5,
                 gate: float = 2.5,
                 max_size_ratio: float = 2.0,
                 max_coast: int = 4,
                 amb_margin: float = 0.05):
        """
        Args:
            target_side: which of the person's arms controls the drone, in
                ANATOMICAL terms ('right' or 'left') regardless of mirroring.
            mirrored: True if the frame has been horizontally flipped (selfie view).
                Pose and Hands disagree about sides under mirroring; see
                _resolve_sides(). The raw Tello feed is not mirrored.
            dist_thresh: max hand-to-skeleton distance, in shoulder-width units.
                Scale-free, so it holds at any distance from the camera.
            side_margin: the target arm must beat the other arm by this factor.
                Rejects ambiguous cases (hands clasped together) instead of guessing.
            face_frac_thresh: fraction of pose face landmarks that must fall inside
                auth_bbox for a skeleton to be considered the authorized person's.
            handedness_conf: only let a MediaPipe handedness label veto a candidate
                when the label is at least this confident.
            num_poses: skeletons to detect per frame. Must be >1 for a bystander
                in frame not to crowd out the operator's skeleton.
            require_arm_visible: reject unless the target arm's elbow and wrist
                lie inside the image with Pose visibility >= arm_vis_thresh. Pose
                always returns every landmark and invents an out-of-frame arm
                from the visible hand, so without this check a hand with no arm
                in shot still "matches" its own guessed arm.
            arm_vis_thresh: minimum Pose visibility for that check.
            gate: how far the palm anchor may move between frames, in palm sizes.
                Scale-free, so it holds at any distance from the camera; it widens
                while the track coasts, since the hand keeps moving unseen.
            max_size_ratio: reject a candidate whose palm grew or shrank by more
                than this factor. A hand does not change size in one frame; a
                different hand usually does.
            max_coast: frames to hold the track with no acceptable candidate
                before declaring it lost.
            amb_margin: DIoU gap below which the two best candidates are too close
                to call, so pose re-association decides instead of the tracker.
        """
        path = Path(model_path) if model_path else DEFAULT_POSE_MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"pose model not found at {path}; download pose_landmarker_full.task "
                "from https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=path.read_bytes()),
            num_poses=int(num_poses),
            output_segmentation_masks=False)
        self._pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        self.model_path = str(path)
        self.num_poses = int(num_poses)
        self._configure(target_side, mirrored, dist_thresh, side_margin,
                        face_frac_thresh, handedness_conf,
                        require_arm_visible, arm_vis_thresh,
                        gate, max_size_ratio, max_coast, amb_margin)

    def _configure(self, target_side, mirrored, dist_thresh, side_margin,
                   face_frac_thresh, handedness_conf,
                   require_arm_visible=True, arm_vis_thresh=0.5,
                   gate=2.5, max_size_ratio=2.0, max_coast=4, amb_margin=0.05):
        # 1-6 eyes, 9-10 mouth corners: the pose face points checked against the box
        self._face_landmarks = {1, 2, 3, 4, 5, 6, 9, 10}
        # 11 - left shoulder, 12 - right shoulder (the scale reference)
        self._shoulders = (11, 12)
        # pose arm end-points, keyed for readability
        self._right_hand_landmarks = {'right_wrist': 16, 'right_pinky': 18,
                                      'right_index': 20, 'right_thumb': 22}
        self._left_hand_landmarks = {'left_wrist': 15, 'left_pinky': 17,
                                     'left_index': 19, 'left_thumb': 21}
        # shoulder, elbow, wrist per pose side
        self._arm_chain = {'right': (12, 14, 16), 'left': (11, 13, 15)}
        self.require_arm_visible = require_arm_visible
        self.arm_vis_thresh = arm_vis_thresh
        # the most recent pose pass, for debug drawing
        self.last_pose = None
        self.last_skeleton = None
        self.last_pose_t = None
        self.target_side = target_side
        self.mirrored = mirrored
        self.dist_thresh = dist_thresh
        self.side_margin = side_margin
        self.face_frac_thresh = face_frac_thresh
        self.handedness_conf = handedness_conf
        self.gate = gate
        self.max_size_ratio = max_size_ratio
        self.max_coast = int(max_coast)
        self.amb_margin = amb_margin
        # the palm being carried between pose passes; see start_track()
        self._track = None
        # Why the last associate_hand() call decided what it did, for the logs.
        self.last_info = {}

    # ------------------------------------------- tracker geometry (frame pixels)
    @staticmethod
    def _to_px(lm, img_shape):
        H, W = img_shape[:2]
        return np.asarray(lm, dtype=np.float32)[:, :2] * (W, H)   # (21, 2)

    @staticmethod
    # Resulting anchor point for the hand, typically the palm center.
    def _anchor(pts):
        return pts[list(PALM_IDX)].mean(axis=0)

    @staticmethod
    def _palm_bbox(pts):
        """The hand's box from wrist + MCPs only: the part that does not deform
        as the fingers move, so a gesture change does not look like a new hand."""
        p = pts[list(PALM_IDX)]
        return np.concatenate([p.min(axis=0), p.max(axis=0)])   # x1, y1, x2, y2

    @staticmethod
    def _palm_size(pts):
        return max(float(np.linalg.norm(pts[0] - pts[9])),      # wrist -> middle MCP
                   float(np.linalg.norm(pts[5] - pts[17])),     # index MCP -> pinky MCP
                   1.0)

    @staticmethod
    # cost function for comparing two hand bounding boxes
    def _diou(a, b):
        """Distance-IoU: IoU penalised by centre separation, so it still ranks
        boxes that do not overlap at all -- which plain IoU cannot."""
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = iw * ih
        union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9
        iou = inter / union
        # distance between the centers of the two boxes squared
        d2 = (((a[0]+a[2]) - (b[0]+b[2])) / 2) ** 2 + (((a[1]+a[3]) - (b[1]+b[3])) / 2) ** 2
        # diagonal of the smallest enclosing box squared (between top leftmost conrer and bottom rightmost corner)
        c2 = (max(a[2], b[2]) - min(a[0], b[0])) ** 2 + (max(a[3], b[3]) - min(a[1], b[1])) ** 2 + 1e-9
        return iou - d2 / c2

    def _make_track(self, pts):
        return _HandTrack(self._anchor(pts), self._palm_bbox(pts), self._palm_size(pts))

    def detect(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._pose_landmarker.detect(image)

    def _resolve_sides(self):
        """Maps the anatomical target_side onto the two models' side conventions.

        MediaPipe Hands labels handedness *assuming the image is already mirrored*
        (selfie camera), so on a mirrored frame its label is anatomically correct.
        MediaPipe Pose labels sides from what it sees, so on a mirrored frame it
        calls the person's real right arm 'left'. The two therefore disagree, and
        exactly one of them needs flipping depending on `mirrored`.

        Returns:
            (pose_side, other_pose_side, expected_hands_label)
        """
        target = 'right' if self.target_side == 'right' else 'left'
        opposite = 'left' if target == 'right' else 'right'
        if self.mirrored:
            return opposite, target, target
        return target, opposite, opposite

    def _pose_scale(self, pose_landmarks, auth_bbox=None, img_shape=None) -> float:
        """Person size in normalized x/y units, used to make distances scale-free.

        Shoulder width is the natural choice: it is roughly constant per person and
        directly comparable to arm-length distances. Falls back to the face box
        diagonal when the shoulders collapse (person side-on or partly out of frame).
        """
        a, b = self._shoulders
        if pose_landmarks is not None and len(pose_landmarks) > max(a, b):
            la, lb = pose_landmarks[a], pose_landmarks[b]
            width = float(np.hypot(la.x - lb.x, la.y - lb.y))
            if width >= 1e-3:
                return width

        if auth_bbox is not None and img_shape is not None:
            h, w = img_shape[:2]
            diag = float(np.hypot(auth_bbox[2] / w, auth_bbox[3] / h))
            if diag >= 1e-3:
                return diag

        return 1e-3  # degenerate; every distance becomes huge and gets rejected

    def find_skeleton(self, pose_result, auth_bbox, img_shape):
        """Finds which detected pose belongs to the authorized face.

        Scores every pose by how many of its face landmarks fall inside auth_bbox
        and returns the best one, provided it clears `face_frac_thresh`. A
        fractional gate rather than all-or-nothing: detector boxes are tight, so
        the mouth corners routinely sit just outside an otherwise perfect match.

        Returns:
            (index, hits) of the best pose, or (None, best_hits) if none qualifies.
        """
        if not pose_result or not pose_result.pose_landmarks:
            return None, 0

        # auth_bbox is in pixel coords (origin_x, origin_y, width, height); pose
        # landmarks are normalized to [0, 1], so convert to normalized (x1,y1,x2,y2).
        h, w = img_shape[:2]
        bx1 = auth_bbox[0] / w
        by1 = auth_bbox[1] / h
        bx2 = (auth_bbox[0] + auth_bbox[2]) / w
        by2 = (auth_bbox[1] + auth_bbox[3]) / h

        best_idx, best_hits = None, 0
        needed = self.face_frac_thresh * len(self._face_landmarks)
        for idx, pose_landmarks in enumerate(pose_result.pose_landmarks):
            hits = 0
            for lm in self._face_landmarks:
                curr_lm = pose_landmarks[lm]
                if bx1 <= curr_lm.x <= bx2 and by1 <= curr_lm.y <= by2:
                    hits += 1
            if hits > best_hits:
                best_hits = hits
                best_idx = idx

        if best_idx is None or best_hits < needed:
            return None, best_hits
        return best_idx, best_hits

    def score_candidate(self, hand_lm, pose_landmarks, scale, label=None, score=None):
        """Scores one candidate hand against one skeleton.

        Returns:
            (d_target, d_other, accepted, reason) where the distances are in
            shoulder-width units and `reason` explains a rejection (or 'ok').
        """
        pose_side, other_side, expected_label = self._resolve_sides()
        d_target = self.hand_against_skeleton(hand_lm, pose_landmarks, pose_side, scale)
        d_other = self.hand_against_skeleton(hand_lm, pose_landmarks, other_side, scale)

        # A confident handedness label that contradicts the target arm is a veto.
        # An unconfident one is ignored: MediaPipe's label degrades badly on
        # occluded or strongly rotated hands, and geometry is the better signal there.
        if (label is not None and score is not None
                and score >= self.handedness_conf and label.lower() != expected_label):
            return d_target, d_other, False, f"handedness={label}({score:.2f})"

        if d_target > self.dist_thresh:
            return d_target, d_other, False, f"far({d_target:.2f}>{self.dist_thresh:.2f})"

        # The hand must be clearly nearer the target arm than the other arm. When
        # the hands are together neither wins, and returning None beats guessing.
        if not d_target < self.side_margin * d_other:
            return d_target, d_other, False, f"ambiguous({d_target:.2f}vs{d_other:.2f})"

        return d_target, d_other, True, "ok"

    def hand_against_skeleton(self, hand_landmarks, pose_landmarks, side='right', scale=1.0):
        """Mean distance between a hand's anchor points and one arm of a skeleton.

        Compares x/y only. The two models' z axes are not the same quantity --
        MediaPipe Hands reports depth relative to that hand's own wrist, MediaPipe
        Pose relative to the hip midpoint -- so including z adds a large residual
        that is near-identical for both arms and drowns out the x/y signal that
        actually tells them apart.

        Returns:
            Mean per-anchor distance in shoulder-width units. On a real
            multi-person photo: ~0.22 for the correct arm vs ~2.3 for the opposite
            arm and ~4.2 for another person's hand -- roughly a 10x margin, which
            is what puts dist_thresh at 0.5.
        """
        wrist_lm, thumb_lm, index_lm, pinky_lm = 0, 4, 8, 20

        group = self._right_hand_landmarks if side == 'right' else self._left_hand_landmarks
        prefix = side
        pairs = (
            (wrist_lm, group[f'{prefix}_wrist']),
            (thumb_lm, group[f'{prefix}_thumb']),
            (index_lm, group[f'{prefix}_index']),
            (pinky_lm, group[f'{prefix}_pinky']),
        )
        # take the sum of normalized distances between hand and pose anchor points
        total = 0.0
        for hand_i, pose_i in pairs:
            p = pose_landmarks[pose_i]
            total += float(np.hypot(hand_landmarks[hand_i][0] - p.x,
                                    hand_landmarks[hand_i][1] - p.y))
        return (total / len(pairs)) / max(scale, 1e-6)

    def arm_visibility(self, pose_landmarks):
        """Whether the target arm's elbow and wrist are really in shot.

        Returns:
            (ok, failing_joint) where failing_joint is 'elbow' or 'wrist' when
            that joint is outside the image or below arm_vis_thresh.
        """
        pose_side, _, _ = self._resolve_sides()
        _, elbow, wrist = self._arm_chain[pose_side]
        for name, i in (("elbow", elbow), ("wrist", wrist)):
            p = pose_landmarks[i]
            vis = getattr(p, "visibility", None)
            in_frame = 0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0
            if not in_frame or vis is None or vis < self.arm_vis_thresh:
                return False, name
        return True, ""

    # ---------------------------------------------------------------- selection
    def associate_hand(self, bgr, auth_bbox, hand_detections) -> HandDetection | None:
        """Associates one hand, across all detections, with the authorized face.

        auth_bbox: (origin_x, origin_y, width, height) in pixels, the verified face.
        hand_detections: list of HandDetection; every hand in each is a candidate.

        Picks the pose whose face landmarks fall inside auth_bbox, then the hand
        nearest that skeleton's target arm. Expensive (full pose inference), so it
        is meant to run periodically, with update_hand_from_previous() in between.

        Returns:
            A single-hand HandDetection, or None if no hand qualifies.
        """
        info = {"skeleton_hits": 0, "n_candidates": 0, "d_target": None, "reason": ""}
        self.last_info = info
        if bgr is None or auth_bbox is None:
            info["reason"] = "no_face"
            return None

        pose_result = self.detect(bgr)
        skeleton, hits = self.find_skeleton(pose_result, auth_bbox, bgr.shape)
        self.last_pose, self.last_skeleton = pose_result, skeleton
        self.last_pose_t = time.time()
        info["skeleton_hits"] = hits
        if skeleton is None:
            info["reason"] = "no_skeleton"
            return None

        pose_landmarks = pose_result.pose_landmarks[skeleton]
        if self.require_arm_visible:
            ok, joint = self.arm_visibility(pose_landmarks)
            if not ok:
                info["reason"] = f"arm_not_visible({joint})"
                return None
        scale = self._pose_scale(pose_landmarks, auth_bbox, bgr.shape)

        # Evaluate every hand in every detection, keep the best-scoring survivor.
        min_dist = float('inf')
        best = None
        reasons = []
        for det in hand_detections:
            for lm, label, conf in _candidates(det):
                info["n_candidates"] += 1
                d_target, _, accepted, reason = self.score_candidate(
                    lm, pose_landmarks, scale, label, conf)
                reasons.append(reason)
                if accepted and d_target < min_dist:
                    min_dist = d_target
                    best = (lm, label, conf)

        if best is not None:
            info["d_target"] = min_dist
            info["reason"] = "ok"
            return _single(*best)
        info["reason"] = ";".join(reasons) if reasons else "no_hand"
        return None

    # ----------------------------------------------------------- track lifecycle
    def start_track(self, hand_det, img_shape=None):
        """Call right after associate_hand() (and whenever the association changes)."""
        if hand_det is None or not hand_det.has_hand or hand_det.landmarks is None:
            self.clear_track()
            return
        self._track = self._make_track(self._to_px(hand_det.landmarks, img_shape))

    def clear_track(self):
        self._track = None

    def update_hand_from_previous(self, hand_detections, img_shape) -> tuple[HandDetection | None, str]:
        """Match the tracked hand against current detections (gate, then DIoU rank).

        Returns:
            (hand, status), status in {"tracked", "coasting", "lost"}.
            "tracked":  hand is a single-hand detection, use it.
            "coasting": hand is None, hold gestures, do NOT re-associate yet.
            "lost":     hand is None, call associate_hand() then start_track().
        """
        tr = self._track
        if tr is None:
            return None, "lost"
        # gate for how far the hand can move relative to the previous track before being considered lost
        # widen it while coasting
        gate = self.gate * (1.0 + 0.5 * tr.missed)   
        scored = []
        for det in hand_detections:
            for lm, label, conf in _candidates(det):
                pts = self._to_px(lm, img_shape)
                anchor = self._anchor(pts)
                size = self._palm_size(pts)
                # skip candidates whose anchor point moved too far from the previous track
                if np.linalg.norm(anchor - tr.anchor) / tr.size > gate:
                    continue
                ratio = size / tr.size
                if not (1.0 / self.max_size_ratio <= ratio <= self.max_size_ratio):
                    continue

                diou = self._diou(tr.bbox, self._palm_bbox(pts))
                scored.append((diou, pts, (lm, label, conf)))

        # no valid candidate: hold the previous state for a few frames
        if not scored:
            tr.missed += 1
            if tr.missed > self.max_coast:
                self.clear_track()
                return None, "lost"
            return None, "coasting"

        scored.sort(key=lambda s: s[0], reverse=True)

        # two candidates too close to call (ambiguous): let pose re-association decide
        if len(scored) > 1 and scored[0][0] - scored[1][0] < self.amb_margin:
            self.clear_track()
            return None, "lost"

        _, pts, best = scored[0]
        self._track = self._make_track(pts)   # state = latest measurement, missed reset to 0
        return _single(*best), "tracked"


@dataclass
class AssocStep:
    """What one tracker step decided, for the per-frame logs."""
    mode: str = ""          # pose | track | coast | reacquire | held | "" (not attempted)
    ok: bool = False
    reason: str = ""
    n_hands: int = 0
    skeleton_hits: int = 0
    d_target: Optional[float] = None


class AssociationTracker:
    """Per-frame association state shared by the controller and the webcam harness.

    Pose inference runs on the first sighting and then every `every_n` fresh hand
    detections; between those, the hand is carried by its palm track. A track with
    nothing acceptable to match coasts, holding the last hand for up to
    `max_coast` frames; only once it is lost does pose re-run on the same frame.
    On frames where the hand detector did not run (it is throttled), the last
    result is held, so pose is never compared against a stale hand.
    """

    def __init__(self, associator: HandAssociator, every_n: int = 20):
        self.associator = associator
        # 0 runs pose on every fresh hand detection (live mode, for debugging);
        # it costs a pose inference per detection instead of one per ~20.
        self.every_n = max(int(every_n), 0)
        self._prev: Optional[HandDetection] = None
        self._since_pose = 0
        self.last = AssocStep()

    def reset(self) -> None:
        self._prev = None
        self._since_pose = 0
        self.associator.clear_track()

    def step(self, frame, det, auth_bbox, fresh: bool) -> Optional[HandDetection]:
        n_hands = len(_candidates(det))
        if auth_bbox is None or n_hands == 0:
            self.reset()
            self.last = AssocStep(reason="no_face" if auth_bbox is None else "no_hand",
                                  n_hands=n_hands)
            return None

        if not fresh:
            self.last = AssocStep(mode="held", ok=self._prev is not None,
                                  reason="held", n_hands=n_hands)
            return self._prev

        a = self.associator
        reason = None
        # re-run pose association if there is no previous hand or the periodic interval has been reached
        if self._prev is None or self._since_pose >= self.every_n:
            got, mode = a.associate_hand(frame, auth_bbox, [det]), "pose"
            a.start_track(got, frame.shape)   # clears the track when got is None
            self._since_pose = 0
        else:
            got, status = a.update_hand_from_previous([det], frame.shape)
            self._since_pose += 1
            if status == "tracked":
                mode = "track"
            elif status == "coasting":
                # the track is still alive: hold the last hand, do not pay for pose
                got, mode, reason = self._prev, "coast", "coasting"
            else:
                got, mode = a.associate_hand(frame, auth_bbox, [det]), "reacquire"
                a.start_track(got, frame.shape)
                self._since_pose = 0

        info = a.last_info if mode in ("pose", "reacquire") else {}
        self._prev = got
        if reason is None:
            reason = "ok" if got is not None else info.get("reason", "lost")
        self.last = AssocStep(
            mode=mode, ok=got is not None, reason=reason,
            n_hands=n_hands,
            skeleton_hits=int(info.get("skeleton_hits", 0)),
            d_target=info.get("d_target"))
        return got
