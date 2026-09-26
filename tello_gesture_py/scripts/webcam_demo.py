"""webcam_demo.py

Webcam-only, drone-free test harness that mirrors the drone control logic:
- Face-follow ONLY when Face-ID matches enrolled template
- Gestures ONLY when authorized face is present
- Search_360 triggers after no authorized face/hand for N seconds, exits immediately when reacquired

This is the intended way to collect real Table A (FPS/latency) and live
Table B (scenario) evidence without physical drone hardware: it runs the
exact same perception + gating + DeterministicModeManager code as
controller.py, just without opening a Tello UDP connection or sending RC
commands to real motors.

Run from the project root:
    python -m tello_gesture_py.scripts.webcam_demo

Keys:
  p     : start/stop FaceID enrollment (collects N crops)
  o     : clear enrolled face
  t     : toggle "flying" (simulates takeoff; only affects the printed command string)
  l     : set flying False (simulates land)
  1-8   : start a scenario trial (see SCENARIO_KEYS below) -- press while performing the behavior
  y / n : record pass/fail for the currently open scenario trial
  q     : quit (saves perf log + scenario log to outputs/metrics/)

Requires an ONNX face-embedding model at models/third_party/arcface.onnx for
FaceID to actually authorize anyone; without it, FaceID stays disabled and
every gesture is treated as unauthorized (hand_detected forced to False),
which is still useful for confirming that rejection path.
"""

import argparse
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from tello_gesture_py.src.face_follow import FaceFollower
from tello_gesture_py.src.face_id import FaceID, FaceIDConfig
from tello_gesture_py.src.hand_gesture import HandGesture
from tello_gesture_py.src.hand_association import AssociationTracker, HandAssociator
from tello_gesture_py.src.config import ControllerConfig
from tello_gesture_py.src.association_overlay import draw_debug
from tello_gesture_py.src.gesture_logic import RuleBasedGesture, rc_from_gesture_name
from tello_gesture_py.src.mode_manager import DeterministicModeManager, DeterministicConfig
from tello_gesture_py.src.rc_command import RCCommand as RC
from tello_gesture_py.src.perf_logger import PerfLogger, StageTimer
from tello_gesture_py.src.telemetry_logger import ScenarioLogger, DecisionLogger
from tello_gesture_py.src.run_context import RunContext


# ---- tweakables ----
MODEL_PATH = str(PROJECT_ROOT / "models" / "third_party" / "arcface.onnx")
COS_THR = 0.55
ENROLL_SAMPLES = 20

# Matches the runtime config in controller.py, which matches the ARO
# manuscript's reported reproducibility parameters (originally tested config).
NOHUMAN_SEARCH_S = 10.0
SEARCH_DURATION_S = 5.0
SEARCH_COOLDOWN_S = 10.0
MODE_HOLD_S = 1.2
HAND_RELEASE_S = 0.8
FACE_RELEASE_S = 0.8
SEARCH_YAW_CMD = 80  # simulated yaw when "searching"

HAND_EVERY_N = 2
FACE_EVERY_N = 4

RC_SPEED = 40  # speed used for gesture->RC conversion

SCENARIO_KEYS = {
    ord("1"): "authorized_gesture_accepted",
    ord("2"): "unauthorized_gesture_rejected",
    ord("3"): "face_following_activates",
    ord("4"): "gesture_preempts_face",
    ord("5"): "hysteresis_prevents_flicker",
    ord("6"): "search_starts_after_loss",
    ord("7"): "target_reacquired_after_search",
    # battery_failsafe_landing (#8) has no webcam analog -- covered only by
    # src/gestures/simulate_scenarios.py (real drone telemetry required otherwise).
}


def rc_to_command_str(rc: RC, flying: bool) -> str:
    if not flying:
        return "ground"
    return f"rc lr={rc.lr} fb={rc.fb} ud={rc.ud} yaw={rc.yaw}"


def _face_authorized(face_id: FaceID, crop_bgr) -> bool:
    if crop_bgr is None:
        return False
    if hasattr(face_id, "is_authorized"):
        return bool(face_id.is_authorized(crop_bgr))
    score = float(face_id.match_score(crop_bgr))
    return bool(face_id.enrolled and score >= float(face_id.cfg.cosine_thr))


def _assoc_fields(assoc) -> dict:
    """Per-frame association outcome; empty when association is off."""
    if assoc is None:
        return {}
    a = assoc.last
    return {
        "assoc_mode": a.mode,
        "assoc_ok": int(a.ok),
        "assoc_hands": a.n_hands,
        "assoc_skeleton_hits": a.skeleton_hits,
        "assoc_d": None if a.d_target is None else round(float(a.d_target), 3),
        "assoc_reason": a.reason,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m tello_gesture_py.scripts.webcam_demo",
        description="Drone-free harness: same perception, gating and arbitration, on a webcam.",
    )
    ap.add_argument("--run-id", default="webcam", help="Short tag; names the output directory.")
    ap.add_argument("--note", default="", help="Free-text note stored in the run manifest.")
    ap.add_argument("--cam", type=int, default=0, help="Webcam index.")
    ap.add_argument("--cosine-thr", type=float, default=COS_THR, help="FaceID authorization threshold.")
    ap.add_argument("--no-auth-hysteresis", action="store_true",
                    help="Bare per-frame identity threshold (pre-fix behaviour).")
    ap.add_argument("--associate-hands", action="store_true",
                    help="Accept only a hand on the verified person's arm (hand-face association).")
    ap.add_argument("--target-side", choices=("right", "left"), default="right",
                    help="Anatomical arm that commands.")
    ap.add_argument("--no-arm-check", action="store_true",
                    help="Accept a hand even when its arm is out of frame (original behaviour).")
    ap.add_argument("--live-pose", action="store_true",
                    help="Run pose on every frame from the start (toggle with 'b').")
    ap.add_argument("--flip", action="store_true",
                    help="Mirror the image like a selfie view. Also tells the associator the "
                         "frame is mirrored, so the two cannot disagree.")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Failed to open webcam.")
        return 1

    run = RunContext(run_id=args.run_id, note=args.note)
    print(f"[run] {run.name}")
    print(f"[run] outputs -> {run.dir}")

    cv2.namedWindow("WEBCAM_DEMO", cv2.WINDOW_NORMAL)

    face = FaceFollower()
    # Arbitration holds across gaps; identity crops are gated far more tightly
    # inside FaceFollower (crop_max_age_s), so a stale box is never embedded.
    face.cfg.lost_timeout_s = 2.0
    face.cfg.detect_every_n = 1  # this script already throttles observe()

    face_id = FaceID(FaceIDConfig(
        model_path=MODEL_PATH,
        cosine_thr=args.cosine_thr,
        hysteresis=not args.no_auth_hysteresis,
        enroll_samples=ENROLL_SAMPLES,
    ))
    if not face_id.enabled:
        print(f"[FaceID] DISABLED -- model not found at {MODEL_PATH}. "
              "Every face will be treated as unauthorized.")

    acfg = ControllerConfig()
    hand = HandGesture(max_num_hands=acfg.assoc_max_hands if args.associate_hands else 1)
    assoc = None
    if args.associate_hands:
        assoc = AssociationTracker(HandAssociator(
            target_side=args.target_side, mirrored=args.flip,
            dist_thresh=acfg.assoc_dist_thresh, side_margin=acfg.assoc_side_margin,
            face_frac_thresh=acfg.assoc_face_frac, min_iou=acfg.assoc_min_iou,
            handedness_conf=acfg.assoc_handedness_conf, num_poses=acfg.assoc_num_poses,
            require_arm_visible=not args.no_arm_check, arm_vis_thresh=acfg.assoc_arm_vis),
            every_n=0 if args.live_pose else acfg.assoc_every_n)
    rule = RuleBasedGesture(dir_thr=0.10, scale_thr=0.18, ema_alpha=0.35)

    mode_mgr = DeterministicModeManager(DeterministicConfig(
        battery_land_pct=15,          # no real battery signal from a webcam
        nohuman_search_s=NOHUMAN_SEARCH_S,
        search_duration_s=SEARCH_DURATION_S,
        search_cooldown_s=SEARCH_COOLDOWN_S,
        mode_hold_s=MODE_HOLD_S,
        hand_release_s=HAND_RELEASE_S,
        face_release_s=FACE_RELEASE_S,
    ))

    perf = PerfLogger(path=run.path("perf.csv"))
    decisions = DecisionLogger(path=run.path("decisions.csv"))
    last_dec_ts = 0.0
    scenarios = ScenarioLogger(path=run.path("scenarios.csv"), run_id=run.name)

    run.record("harness", "webcam_demo")
    run.record("params", {
        "cosine_thr": args.cosine_thr,
        "auth_hysteresis": not args.no_auth_hysteresis,
        "enroll_samples": ENROLL_SAMPLES,
        "nohuman_search_s": NOHUMAN_SEARCH_S,
        "search_duration_s": SEARCH_DURATION_S,
        "mode_hold_s": MODE_HOLD_S,
        "hand_release_s": HAND_RELEASE_S,
        "face_release_s": FACE_RELEASE_S,
        "hand_every_n": HAND_EVERY_N,
        "face_every_n": FACE_EVERY_N,
        "rc_speed": RC_SPEED,
        "cam": args.cam,
        "flip": bool(args.flip),
        "hand_association": bool(args.associate_hands),
        "assoc_target_side": args.target_side,
        "assoc_mirrored": bool(args.flip),
    })
    if assoc is not None:
        run.record("hand_association", {
            "pose_model": assoc.associator.model_path,
            "num_poses": assoc.associator.num_poses,
            "every_n": assoc.every_n,
        })
    run.record("model", {
        "faceid_model": MODEL_PATH,
        "faceid_enabled": bool(face_id.enabled),
        "classifier": "rule_based",
    })

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
    debug_overlay = True

    print("Keys: p=enroll o=clear t=fly l=land  1-7=start scenario trial  y/n=pass/fail  q=quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if args.flip:
                frame = cv2.flip(frame, 1)

            now = time.time()
            timer = StageTimer()

            # ---- Hand detect (throttled) ----
            hand_frame_i += 1
            run_hand = (hand_frame_i % HAND_EVERY_N == 0)
            if run_hand:
                det = hand.detect(frame)
                last_hand_det = det
            else:
                det = last_hand_det
            timer.mark("hand_detect_ms", ran=run_hand)

            raw_hand = bool(det is not None and det.has_hand and det.landmarks is not None)
            hand_streak = min(hand_streak + 1, 10) if raw_hand else 0
            hand_detected_raw = hand_streak >= hand_streak_on

            # ---- Face observe (throttled) ----
            face_frame_i += 1
            run_face = (face_frame_i % FACE_EVERY_N == 0)
            if run_face:
                face.observe(frame)

            raw_face = bool(face.face_detected())
            timer.mark("face_detect_ms", ran=run_face)

            face_crop = face.crop_face(frame) if raw_face else None

            # Enrollment samples come from ANY detected face crop
            if face_id.enrolling and face_crop is not None:
                face_id.add_sample(face_crop)

            authorized_face = _face_authorized(face_id, face_crop)
            timer.mark("face_id_ms", ran=bool(
                face_crop is not None and face_id.enabled
                and (face_id.enrolled or face_id.enrolling)))

            # Streak gating on AUTH face only
            face_streak = min(face_streak + 1, 10) if authorized_face else 0
            face_detected = face_streak >= face_streak_on

            # Gate hand on auth face
            hand_detected = bool(hand_detected_raw and face_detected)

            # ---- Hand-face association (same code as the flight controller) ----
            assoc_det = None
            if assoc is not None:
                t_assoc = time.perf_counter()
                auth_bbox = face.get_last_bbox() if face_detected else None
                assoc_det = assoc.step(frame, det, auth_bbox, fresh=run_hand)
                if assoc.last.mode in ("pose", "reacquire"):
                    timer.record("assoc_pose_ms", (time.perf_counter() - t_assoc) * 1000.0)
                hand_detected = bool(hand_detected and assoc_det is not None)
                if assoc_det is not None:
                    det = assoc_det
                timer.mark("assoc_ms", ran=assoc.last.mode in ("pose", "iou", "reacquire"))

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
            timer.mark("mode_manager_ms")

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

            timer.mark("command_exec_ms")
            command_str = rc_to_command_str(rc, flying)
            camera_to_command_ms = timer.elapsed_ms()

            perf.log_frame(timer, camera_to_command_ms, extra={"mode": mode, **_assoc_fields(assoc)})

            bb = face.get_last_bbox()
            if (now - last_dec_ts) >= 0.5:
                nn, NN = face_id.enroll_progress()
                decisions.add({
                    "mode": mode,
                    "command": command_str,
                    "det_reason": det_reason,
                    "gesture": gesture_name,
                    "face_raw": bool(raw_face),
                    "face_auth": bool(face_detected),
                    "hand_raw": bool(hand_detected_raw),
                    "hand_auth": bool(hand_detected),
                    "faceid_enrolled": bool(face_id.enrolled),
                    "faceid_enrolling": bool(face_id.enrolling),
                    "faceid_progress": f"{nn}/{NN}",
                    "faceid_score": float(face_id.last_score),
                    "face_bbox_w": (bb or (0, 0, None, None))[2],
                    "face_bbox_h": (bb or (0, 0, None, None))[3],
                    "crop_ok": bool(face_crop is not None),
                    "flying": bool(flying),
                    **_assoc_fields(assoc),
                })
                last_dec_ts = now

            # -------- HUD overlay --------
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

            if assoc is not None:
                if debug_overlay:
                    draw_debug(frame, last_hand_det, assoc_det, assoc.associator,
                               face.get_last_bbox(), face_detected)
                a = assoc.last
                d = "" if a.d_target is None else f" d={a.d_target:.2f}"
                hud(f"ASSOC: {'OK' if a.ok else '--'} {a.mode or '-'}{d}  hands={a.n_hands}"
                    f"  side={args.target_side}  flip={'Y' if args.flip else 'N'}  {a.reason[:34]}")
            hud(f"GESTURE: {gesture_name}")
            hud(f"CMD: {command_str}")
            hud("KEYS: p=enroll o=clear  1-7=trial  y=pass n=fail x=discard  q=quit")
            if assoc is not None:
                hud(f"DEBUG: v=landmarks {'ON' if debug_overlay else 'off'}   "
                    f"b=live pose {'ON (every frame)' if assoc.every_n == 0 else 'off'}")

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

            if key == ord("v"):
                debug_overlay = not debug_overlay
            if key == ord("b") and assoc is not None:
                assoc.every_n = 0 if assoc.every_n != 0 else acfg.assoc_every_n
                assoc.reset()
                print(f"[assoc] live pose {'ON' if assoc.every_n == 0 else 'off'}")

            if key == ord("o"):
                face_id.clear()
                print("[FaceID] Cleared enrolled template.")

            if key in SCENARIO_KEYS:
                scenarios.start(SCENARIO_KEYS[key], mode_at_start=mode)
            elif key == ord("y"):
                scenarios.resolve(True, mode_at_end=mode)
            elif key == ord("n"):
                scenarios.resolve(False, mode_at_end=mode)
            elif key == ord("x"):
                scenarios.discard()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        for name, logger in (("perf", perf), ("decisions", decisions), ("scenarios", scenarios)):
            try:
                logger.export()
            except Exception as e:
                print(f"{name} export failed:", e)

        try:
            run.record("summary", {
                "frames_logged": len(perf.rows),
                "decisions": len(decisions.rows),
                "scenario_trials": len(scenarios.rows),
                "scenario_tally": scenarios.tally(),
            })
            run.write()
        except Exception as e:
            print("manifest write failed:", e)

        print("")
        print(f"[run] {run.name} complete")
        print(f"[run] {len(perf.rows)} frames, {len(scenarios.rows)} scenario trials")
        for k, v in sorted(scenarios.tally().items()):
            print(f"        {k:38s} {v}")
        print(f"[run] saved to {run.dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
