"""Debug overlay: every landmark the hand-face association decision reads.

Shared by the flight controller and webcam_demo, so both windows draw the same
thing. Display only: it never feeds back into a decision, and it is drawn after
the frame's latency is logged, so it does not inflate measured latency.
"""
import time

import cv2
import mediapipe as mp
import numpy as np

GREEN, RED, YELLOW = (0, 220, 0), (0, 0, 255), (0, 220, 255)
CYAN, ORANGE, GREY, WHITE = (255, 220, 0), (0, 140, 255), (150, 150, 150), (255, 255, 255)
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
# head, shoulders, arms, torso: the legs play no part in the decision
POSE_CONNECTIONS = [c for c in mp.solutions.pose.POSE_CONNECTIONS if max(c) <= 24]


def _px(p, w, h):
    return int(p.x * w), int(p.y * h)


def _inside(p):
    return 0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0


def draw_debug(frame, raw_det, chosen, associator, face_bbox, face_ok) -> None:
    """Draw the face box, hands, and (with association on) the pose skeleton.

    Face: the verified box, green when authorized and orange when not, plus
    Pose's eye and mouth points that must fall inside it. Hands: 21 landmarks
    each, the accepted hand green, every other hand red. Arm: the matched
    skeleton, commanding arm thick yellow, its shoulder/elbow/wrist ringed
    green or red by the in-frame and visibility check with the visibility
    printed; a joint outside the image is pinned to the edge and labelled.
    Other people's skeletons are grey.

    associator may be None (association off): then only the face box and the
    hands are drawn, all hands in white since nothing selects among them.
    """
    h, w = frame.shape[:2]

    if face_bbox is not None:
        x, y, bw, bh = face_bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), GREEN if face_ok else ORANGE, 2)

    pose = None if associator is None else associator.last_pose
    if pose is not None and pose.pose_landmarks:
        target_side, _, _ = associator._resolve_sides()
        shoulder_i, elbow_i, wrist_i = associator._arm_chain[target_side]
        target_hand = {15, 17, 19, 21} if target_side == "left" else {16, 18, 20, 22}
        arm = {shoulder_i, elbow_i, wrist_i} | target_hand
        for k, pl in enumerate(pose.pose_landmarks):
            mine = k == associator.last_skeleton
            for a, b in POSE_CONNECTIONS:
                if not (_inside(pl[a]) and _inside(pl[b])):
                    continue
                on_arm = {a, b} <= arm
                col = (YELLOW if on_arm else WHITE) if mine else GREY
                cv2.line(frame, _px(pl[a], w, h), _px(pl[b], w, h), col,
                         4 if (mine and on_arm) else 1)
            if not mine:
                continue
            for i in associator._face_landmarks:
                if _inside(pl[i]):
                    cv2.circle(frame, _px(pl[i], w, h), 3, CYAN, -1)
            for name, i in (("shoulder", shoulder_i), ("elbow", elbow_i), ("wrist", wrist_i)):
                p = pl[i]
                vis = getattr(p, "visibility", None) or 0.0
                ok = _inside(p) and vis >= associator.arm_vis_thresh
                cx = min(max(int(p.x * w), 8), w - 8)
                cy = min(max(int(p.y * h), 8), h - 8)
                cv2.circle(frame, (cx, cy), 10, GREEN if ok else RED, 2)
                label = f"{name} {vis:.2f}" + ("" if _inside(p) else " OFF-FRAME")
                cv2.putText(frame, label, (min(cx + 12, w - 170), max(cy - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN if ok else RED, 1, cv2.LINE_AA)
    if associator is not None and associator.last_pose_t is not None:
        age = time.time() - associator.last_pose_t
        cv2.putText(frame, f"pose {age:.1f}s ago", (w - 150, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)

    if raw_det is not None and raw_det.has_hand:
        chosen_lm = None if chosen is None else chosen.landmarks
        for lm in raw_det.hands or [raw_det.landmarks]:
            if associator is None:
                col, is_chosen = WHITE, False
            else:
                is_chosen = chosen_lm is not None and np.array_equal(lm, chosen_lm)
                col = GREEN if is_chosen else RED
            pts = [(int(q[0] * w), int(q[1] * h)) for q in lm]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], col, 2 if is_chosen else 1)
            for q in pts:
                cv2.circle(frame, q, 3, col, -1)
            if is_chosen:
                cv2.putText(frame, "OPERATOR", (pts[0][0] - 30, pts[0][1] + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2, cv2.LINE_AA)
