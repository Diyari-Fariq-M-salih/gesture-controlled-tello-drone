"""
webcam_demo.py

Webcam-only test harness that mirrors the drone logic:
- Face-follow ONLY when Face-ID matches enrolled template
- Gestures ONLY when authorized face is present
- Search_360 triggers after no authorized face/hand for N seconds, exits immediately when reacquired

Keys:
  p : start/stop FaceID enrollment (collects N crops)
  o : clear enrolled face
  t : toggle "flying" (simulates takeoff)
  l : set flying False (simulates land)
  q : quit

Requires your project modules:
  tello_gesture/face_follow.py
  tello_gesture/face_id.py
  tello_gesture/mode_manager.py
  tello_gesture/hand_gesture.py
  tello_gesture/gesture_logic.py
  tello_gesture/rc_command.py

And an ONNX face-embedding model at:
  models/arcface.onnx   (or edit MODEL_PATH below)
"""

import time
import cv2

from tello_gesture.face_follow import FaceFollower
from tello_gesture.face_id import FaceID, FaceIDConfig
from tello_gesture.hand_gesture import HandGesture
from tello_gesture.gesture_logic import RuleBasedGesture, rc_from_gesture_name
from tello_gesture.mode_manager import DeterministicModeManager, DeterministicConfig
from tello_gesture.rc_command import RCCommand as RC


# ---- tweakables ----
MODEL_PATH = "models/arcface.onnx"
COS_THR = 0.55
ENROLL_SAMPLES = 20

NOHUMAN_SEARCH_S = 10.0
SEARCH_DURATION_S = 5.0
SEARCH_COOLDOWN_S = 10.0
SEARCH_YAW_CMD = 80  # simulated yaw when "searching"

HAND_EVERY_N = 2
FACE_EVERY_N = 4

RC_SPEED = 40  # speed used for gesture->RC conversion


def rc_to_command_str(rc: RC, flying: bool) -> str:
    if not flying:
        return "ground"
    return f"rc lr={rc.lr} fb={rc.fb} ud={rc.ud} yaw={rc.yaw}"


def _face_authorized(face_id: FaceID, crop_bgr) -> bool:
    """
    Compatibility helper:
    - If FaceID has is_authorized(), use it.
    - Else use match_score() + threshold.
    """
    if crop_bgr is None:
        return False
    if hasattr(face_id, "is_authorized"):
        return bool(face_id.is_authorized(crop_bgr))
    score = float(face_id.match_score(crop_bgr))
    return bool(face_id.enrolled and score >= float(face_id.cfg.cosine_thr))


def main() -> int:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return 1

    cv2.namedWindow("WEBCAM_DEMO", cv2.WINDOW_NORMAL)

    face = FaceFollower()
    face.cfg.lost_timeout_s = 2.0  # keep face "fresh" longer on low fps

    face_id = FaceID(FaceIDConfig(
        model_path=MODEL_PATH,
        cosine_thr=COS_THR,
        enroll_samples=ENROLL_SAMPLES,
        # If your model expects BGR, set rgb=False in your face_id.py config.
        # rgb=True,
    ))

    hand = HandGesture(max_num_hands=1)
    rule = RuleBasedGesture(dir_thr=0.10, scale_thr=0.18, ema_alpha=0.35)

    mode_mgr = DeterministicModeManager(DeterministicConfig(
        battery_land_pct=15,          # ignored here (no battery)
        nohuman_search_s=NOHUMAN_SEARCH_S,
        search_duration_s=SEARCH_DURATION_S,
        search_cooldown_s=SEARCH_COOLDOWN_S,
        mode_hold_s=1.2,
        hand_release_s=0.8,
        face_release_s=0.8,
    ))

    flying = False

    # Timers (AUTHORIZED signals only)
    last_hand_ts = time.time()
    last_face_ts = time.time()
    last_any_seen_ts = time.time()

    # Throttling
    hand_frame_i = 0
    last_hand_det = None
    face_frame_i = 0

    # Streak gating
    hand_streak = 0
    face_streak = 0
    hand_streak_on = 2
    face_streak_on = 1

    # Gesture stability gating
    gesture_stable_name = "CENTER"
    gesture_stable_count = 0
    gesture_fb_streak_on = 2

    rc = RC(active=False)
    gesture_name = "NOHAND"

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        now = time.time()

        # ---- Hand detect (throttled) ----
        hand_frame_i += 1
        if hand_frame_i % HAND_EVERY_N == 0:
            det = hand.detect(frame)
            last_hand_det = det
        else:
            det = last_hand_det

        raw_hand = bool(det is not None and det.has_hand and det.landmarks is not None)
        hand_streak = min(hand_streak + 1, 10) if raw_hand else 0
        hand_detected_raw = hand_streak >= hand_streak_on

        # ---- Face observe (throttled) ----
        face_frame_i += 1
        if face_frame_i % FACE_EVERY_N == 0:
            face.observe(frame)

        raw_face = bool(face.face_detected())
        face_crop = face.crop_face(frame) if raw_face else None

        # Enrollment samples come from ANY detected face crop
        if face_id.enrolling and face_crop is not None:
            face_id.add_sample(face_crop)

        authorized_face = _face_authorized(face_id, face_crop)

        # Streak gating on AUTH face only
        face_streak = min(face_streak + 1, 10) if authorized_face else 0
        face_detected = face_streak >= face_streak_on

        # Gate hand on auth face
        hand_detected = bool(hand_detected_raw and face_detected)

        # Update timers (AUTHORIZED only)
        if face_detected:
            last_face_ts = now
            last_any_seen_ts = now
        if hand_detected:
            last_hand_ts = now
            last_any_seen_ts = now

        time_since_hand = now - last_hand_ts
        time_since_face = now - last_face_ts
        time_since_any = now - last_any_seen_ts

        # Reset temporal gesture state when hand is gone long enough
        if (now - last_hand_ts) > float(mode_mgr.cfg.hand_release_s):
            try:
                rule.reset()
            except Exception:
                pass

        state_for_mode = {
            "hand_detected": bool(hand_detected),
            "face_detected": bool(face_detected),
            "time_since_hand_s": float(time_since_hand),
            "time_since_face_s": float(time_since_face),
            "time_since_any_seen_s": float(time_since_any),
            "battery": None,
            "altitude_cm": None,
            "flying": bool(flying),
        }

        mode, det_reason = mode_mgr.tick(state_for_mode)

        # ---- Execute mode -> simulated RC ----
        if mode == "gesture":
            if hand_detected and det is not None and det.landmarks is not None:
                gr = rule.predict(det.landmarks)

                if gr.name == gesture_stable_name:
                    gesture_stable_count = min(gesture_stable_count + 1, 50)
                else:
                    gesture_stable_name = gr.name
                    gesture_stable_count = 1

                gesture_name = gr.name

                parts = gesture_name.split("-") if gesture_name else []
                if ("FORWARD" in parts or "BACK" in parts) and gesture_stable_count < gesture_fb_streak_on:
                    parts = [p for p in parts if p not in ("FORWARD", "BACK")]
                    gesture_name = "-".join(parts) if parts else "CENTER"

                rc = rc_from_gesture_name(gesture_name, RC_SPEED) if flying else RC(active=False)
            else:
                gesture_name = "NOHAND"
                gesture_stable_name = "CENTER"
                gesture_stable_count = 0
                rc = RC(active=True) if flying else RC(active=False)

        elif mode == "face":
            if face_detected:
                cmd, frame = face.update(frame)
                rc = cmd if flying else RC(active=False)
                gesture_name = "FACE"
            else:
                rc = RC(active=True) if flying else RC(active=False)
                gesture_name = "HOVER"

        elif mode == "search_360":
            rc = RC(lr=0, fb=0, ud=0, yaw=SEARCH_YAW_CMD, active=True) if flying else RC(active=False)
            gesture_name = "SEARCH"

        elif mode == "hover":
            rc = RC(active=True) if flying else RC(active=False)
            gesture_name = "HOVER"

        elif mode == "land":
            flying = False
            rc = RC(active=False)
            gesture_name = "LAND"

        else:
            rc = RC(active=True) if flying else RC(active=False)
            gesture_name = "OTHER"

        command_str = rc_to_command_str(rc, flying)

        # -------- HUD overlay (compact & organized) --------
        x = 10
        y = 18
        dy = 16
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.45
        th = 1
        col = (255, 255, 255)

        def hud(line: str):
            nonlocal y
            cv2.putText(frame, line, (x, y), font, fs, col, th, cv2.LINE_AA)
            y += dy

        hud(f"MODE: {mode.upper():<10}  FLY: {'Y' if flying else 'N'}")
        hud(f"FACE: raw={int(raw_face)} auth={int(face_detected)}   HAND: raw={int(hand_detected_raw)} auth={int(hand_detected)}")

        n, N = face_id.enroll_progress() if hasattr(face_id, "enroll_progress") else (0, ENROLL_SAMPLES)
        hud(f"FACE-ID: enrolled={'Y' if face_id.enrolled else 'N'}  enrolling={'Y' if face_id.enrolling else 'N'}  ({n}/{N})"
            f"  score={face_id.last_score:.3f}  thr={face_id.cfg.cosine_thr:.2f}")

        hud(f"GESTURE: {gesture_name}")
        hud(f"CMD: {command_str}")
        hud("KEYS: t=fly  l=land  p=enroll  o=clear  q=quit")

        cv2.imshow("WEBCAM_DEMO", frame)

        # ---- Keys ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("t"):
            flying = not flying
            print("flying:", flying)

        if key == ord("l"):
            flying = False
            print("flying: False")

        if key == ord("p"):
            if not face_id.enrolling:
                face_id.start_enroll()
                print("[FaceID] Enrollment started.")
            else:
                face_id.cancel_enroll()
                print("[FaceID] Enrollment cancelled.")

        if key == ord("o"):
            face_id.clear()
            print("[FaceID] Cleared enrolled template.")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
