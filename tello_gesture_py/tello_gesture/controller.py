import time
import cv2
from typing import Optional
import textwrap

from .config import ControllerConfig
from .tello_udp import TelloUDP
from .state_listener import StateListener
from .latest_frame import LatestFrame
from .video_stream import VideoStream
from .hand_gesture import HandGesture
from .gesture_logic import RuleBasedGesture, rc_from_gesture_name
from .keyboard import rc_from_key, RC
from .telemetry_logger import TelemetryLogger, DecisionLogger
from .face_follow import FaceFollower

from .face_id import FaceID, FaceIDConfig

from .mode_manager import (
    DeterministicModeManager,
    DeterministicConfig,
    LLMReasoner,
    LLMReasonConfig,
)


def _put_text_box(
    img,
    text: str,
    org,
    *,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=0.55,
    thickness=1,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
    alpha=0.55,
    pad=4,
):
    """Draw text with a translucent background box for readability."""
    x, y = org
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    x1, y1 = x - pad, y - th - pad
    x2, y2 = x + tw + pad, y + baseline + pad

    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)


def draw_hud(
    frame,
    *,
    mode: str,
    flying: bool,
    hand_detected_raw: bool,
    hand_detected: bool,
    raw_face: bool,
    face_detected: bool,
    gesture_name: str,
    face_id: FaceID,
    llm_reason: Optional[str] = None,
):
    """Centralized overlay/HUD drawing (clean, non-overlapping, readable)."""
    x = 10
    y = 24
    line_h = 24

    # Line 1: status
    _put_text_box(
        frame,
        f"MODE={mode.upper()}  fly={'Y' if flying else 'N'}",
        (x, y),
        scale=0.65,
        thickness=2,
        alpha=0.55,
    )
    y += line_h

    # Line 2: detectors
    _put_text_box(
        frame,
        f"hand raw/auth: {int(hand_detected_raw)}/{int(hand_detected)}   "
        f"face raw/auth: {int(raw_face)}/{int(face_detected)}   "
        f"g: {gesture_name}",
        (x, y),
        scale=0.55,
        thickness=1,
        alpha=0.55,
    )
    y += line_h

    # Line 3: FaceID
    n, N = face_id.enroll_progress()
    _put_text_box(
        frame,
        f"FaceID enr:{'Y' if face_id.enrolled else 'N'} "
        f"enrolling:{'Y' if face_id.enrolling else 'N'} "
        f"({n}/{N}) score:{face_id.last_score:.3f} thr:{face_id.cfg.cosine_thr:.2f}",
        (x, y),
        scale=0.52,
        thickness=1,
        alpha=0.55,
    )
    y += line_h

    # Line 4: keys
    _put_text_box(
        frame,
        "Keys: t takeoff | l land | e emergency | q quit | p enrollFace | o clearFace",
        (x, y),
        scale=0.48,
        thickness=1,
        alpha=0.45,
    )
    y += line_h

    # LLM reason: wrap to multiple lines, cap to avoid covering the whole frame
    if llm_reason:
        wrapped = textwrap.wrap(llm_reason.strip(), width=60)[:3]  # max 3 lines
        for i, line in enumerate(wrapped):
            prefix = "LLM:" if i == 0 else "    "
            _put_text_box(
                frame,
                f"{prefix} {line}",
                (x, y),
                scale=0.52,
                thickness=1,
                alpha=0.55,
            )
            y += line_h


class Controller:
    def __init__(self, cfg: ControllerConfig, model_path: Optional[str] = None, labels_path: Optional[str] = None):
        self.cfg = cfg
        self.tello = TelloUDP(cfg.tello_ip, cfg.cmd_port, cfg.local_cmd_port)
        self.state = StateListener(cfg.state_port)
        self.latest = LatestFrame()

        self.video = VideoStream(
            self.latest,
            f"udp://0.0.0.0:{cfg.video_port}?fifo_size=5000000&overrun_nonfatal=1"
        )

        self.hand = HandGesture(max_num_hands=1)
        self.rule = RuleBasedGesture(cfg.dir_thr, cfg.scale_thr, cfg.ema_alpha)

        self.face = FaceFollower()
        # keep face detections "fresh" longer on low FPS streams
        self.face.cfg.lost_timeout_s = 2.0

        # FaceID (identity lock)
        self.face_id = FaceID(FaceIDConfig(
            model_path="models/arcface.onnx",
            cosine_thr=0.55,
            enroll_samples=20,
            rgb=True,  # flip to False only if your scores are garbage
        ))

        # Deterministic mode manager
        self.mode_mgr = DeterministicModeManager(DeterministicConfig(
            battery_land_pct=15,
            nohuman_search_s=5.0,
            search_duration_s=12.0,
            search_cooldown_s=10.0,
            mode_hold_s=2,
            hand_release_s=0.8,
            face_release_s=0.8,
        ))

        # Faster spin
        self._search_yaw_cmd = 30

        # LLM reasoner (reason-only)
        self.reasoner = LLMReasoner(LLMReasonConfig(
            enabled=True,
            decision_hz=1.0,
            timeout_s=4.0,
        ))

        self.flying = False

        self.logger = TelemetryLogger(fields=["bat", "h", "yaw"], path=cfg.log_path)
        dec_path = cfg.log_path.replace(".csv", "_decisions.csv")
        self.decisions = DecisionLogger(path=dec_path)

        # Timers (AUTHORIZED signals only)
        self._last_hand_ts = time.time()
        self._last_face_ts = time.time()
        self._last_any_seen_ts = time.time()

        # Throttling
        self._hand_frame_i = 0
        self._hand_every_n = 2
        self._last_hand_det = None

        self._face_frame_i = 0
        self._face_every_n = 4

        # Streak gating (light)
        self._hand_streak = 0
        self._face_streak = 0
        self._hand_streak_on = 2
        self._face_streak_on = 1

        # Gesture stability gating
        self._gesture_stable_name = "CENTER"
        self._gesture_stable_count = 0
        self._gesture_fb_streak_on = 2

        # CSV logging control
        self._prev_mode = ""
        self._prev_cmd = ""
        self._prev_llm_reason = ""
        self._last_log_ts = 0.0
        self._log_every_s = 1.0

        self._warned_deadband = False

        self._trained = None
        if model_path and labels_path:
            from .model_classifier import TrainedClassifier
            self._trained = TrainedClassifier(model_path, labels_path)

    def _sdk_init(self) -> bool:
        ok, resp = self.tello.send_cmd("command", timeout_ms=6000)
        if not ok or resp.lower() != "ok":
            print("Failed SDK mode:", ok, resp)
            return False
        self.tello.send_cmd("streamoff", timeout_ms=2000)
        self.tello.send_cmd("streamon", timeout_ms=6000)
        return True

    def _rc_to_command_str(self, rc: RC) -> str:
        if not self.flying:
            return "ground"
        return f"rc lr={rc.lr} fb={rc.fb} ud={rc.ud} yaw={rc.yaw}"

    def _handle_faceid_keys(self, key: int):
        # p = start/stop enrollment
        if key == ord("p"):
            if not self.face_id.enrolling:
                self.face_id.start_enroll()
                print("[FaceID] Enrollment started (need 20 crops).")
            else:
                self.face_id.cancel_enroll()
                print("[FaceID] Enrollment cancelled.")

        # o = clear
        if key == ord("o"):
            self.face_id.clear()
            print("[FaceID] Cleared enrolled template.")

    def run(self) -> int:
        if not self._sdk_init():
            return 1

        self.state.start()
        if not self.video.start():
            print("Video stream not opened. Check firewall UDP 11111.")
        cv2.namedWindow("TELLO", cv2.WINDOW_NORMAL)

        rc = RC(active=False)
        last_rc_send = time.time()
        gesture_name = "NOHAND"

        try:
            while True:
                ok, frame, _, _ = self.latest.get(copy=True)
                now = time.time()

                if ok and frame is not None:
                    # --- Hand detect (throttled) ---
                    self._hand_frame_i += 1
                    if self._hand_frame_i % self._hand_every_n == 0:
                        det = self.hand.detect(frame)
                        self._last_hand_det = det
                    else:
                        det = self._last_hand_det

                    raw_hand = bool(det is not None and det.has_hand and det.landmarks is not None)
                    self._hand_streak = min(self._hand_streak + 1, 10) if raw_hand else 0
                    hand_detected_raw = self._hand_streak >= self._hand_streak_on

                    # --- Face observe (throttled) ---
                    self._face_frame_i += 1
                    if self._face_frame_i % self._face_every_n == 0:
                        self.face.observe(frame)

                    raw_face = bool(self.face.face_detected())

                    # --- FaceID crop + enroll + authorize ---
                    face_crop = self.face.crop_face(frame) if raw_face else None

                    if self.face_id.enrolling and face_crop is not None:
                        self.face_id.add_sample(face_crop)

                    authorized_face = bool(face_crop is not None and self.face_id.is_authorized(face_crop))

                    # streak gating on AUTHORIZED face
                    self._face_streak = min(self._face_streak + 1, 10) if authorized_face else 0
                    face_detected = self._face_streak >= self._face_streak_on

                    # HARD GATE: gesture only if authorized face present
                    hand_detected = bool(hand_detected_raw and face_detected)

                    # Update timers (AUTHORIZED only)
                    if face_detected:
                        self._last_face_ts = now
                        self._last_any_seen_ts = now
                    if hand_detected:
                        self._last_hand_ts = now
                        self._last_any_seen_ts = now

                    # --- Telemetry snapshot ---
                    st = self.state.snapshot()
                    bat = st.get("bat", None)
                    alt = st.get("h", None)

                    time_since_hand = now - self._last_hand_ts
                    time_since_face = now - self._last_face_ts
                    time_since_any = now - self._last_any_seen_ts

                    # Reset temporal gesture state when authorized hand is gone
                    if (now - self._last_hand_ts) > float(self.mode_mgr.cfg.hand_release_s):
                        try:
                            self.rule.reset()
                        except Exception:
                            pass

                    state_for_mode = {
                        "hand_detected": bool(hand_detected),
                        "face_detected": bool(face_detected),
                        "time_since_hand_s": float(time_since_hand),
                        "time_since_face_s": float(time_since_face),
                        "time_since_any_seen_s": float(time_since_any),
                        "battery": bat if bat is None else float(bat),
                        "altitude_cm": alt if alt is None else float(alt),
                        "flying": bool(self.flying),
                    }

                    mode, det_reason = self.mode_mgr.tick(state_for_mode)

                    # --- Execute deterministic mode ---
                    if mode == "gesture":
                        if hand_detected and det is not None and det.landmarks is not None:
                            if self._trained is not None:
                                gr = self._trained.predict(det.landmarks)
                            else:
                                gr = self.rule.predict(det.landmarks)

                            if gr.name == self._gesture_stable_name:
                                self._gesture_stable_count = min(self._gesture_stable_count + 1, 50)
                            else:
                                self._gesture_stable_name = gr.name
                                self._gesture_stable_count = 1

                            gesture_name = gr.name

                            parts = gesture_name.split("-") if gesture_name else []
                            if ("FORWARD" in parts or "BACK" in parts) and self._gesture_stable_count < self._gesture_fb_streak_on:
                                parts = [p for p in parts if p not in ("FORWARD", "BACK")]
                                gesture_name = "-".join(parts) if parts else "CENTER"

                            rc = rc_from_gesture_name(gesture_name, self.cfg.rc_speed) if self.flying else RC(active=False)
                        else:
                            gesture_name = "NOHAND"
                            self._gesture_stable_name = "CENTER"
                            self._gesture_stable_count = 0
                            rc = RC(active=True) if self.flying else RC(active=False)

                    elif mode == "face":
                        if face_detected:
                            cmd, frame = self.face.update(frame)
                            rc = cmd if self.flying else RC(active=False)
                            gesture_name = "FACE"
                        else:
                            rc = RC(active=True) if self.flying else RC(active=False)
                            gesture_name = "HOVER"

                    elif mode == "search_360":
                        rc = RC(lr=0, fb=0, ud=0, yaw=self._search_yaw_cmd, active=True) if self.flying else RC(active=False)
                        gesture_name = "SEARCH"

                    elif mode == "hover":
                        rc = RC(active=True) if self.flying else RC(active=False)
                        gesture_name = "HOVER"

                    elif mode == "land":
                        if self.flying:
                            self.tello.send_cmd("land", timeout_ms=8000)
                        self.flying = False
                        rc = RC(active=False)
                        gesture_name = "LAND"

                    else:
                        rc = RC(active=True) if self.flying else RC(active=False)
                        gesture_name = "OTHER"

                    command_str = self._rc_to_command_str(rc)

                    # --- LLM reason (reason-only) ---
                    llm_payload = {
                        "mode": mode,
                        "command": command_str,
                        "deterministic_reason": det_reason,
                        "hand_detected": bool(hand_detected),
                        "face_detected": bool(face_detected),
                        "battery": bat,
                        "altitude_cm": alt,
                        "time_since_any_seen_s": float(time_since_any),
                        "flying": bool(self.flying),
                    }
                    self.reasoner.tick(llm_payload)
                    llm_reason = self.reasoner.get_reason()

                    # --- Log decisions ---
                    should_log = False
                    if (now - self._last_log_ts) >= self._log_every_s:
                        should_log = True
                    if mode != self._prev_mode or command_str != self._prev_cmd:
                        should_log = True
                    if llm_reason and llm_reason != self._prev_llm_reason:
                        should_log = True

                    if should_log:
                        n, N = self.face_id.enroll_progress()
                        self.decisions.add({
                            "mode": mode,
                            "command": command_str,
                            "llm_reason": llm_reason,
                            "det_reason": det_reason,
                            "battery": bat,
                            "altitude_cm": alt,
                            "face_raw": bool(raw_face),
                            "face_auth": bool(face_detected),
                            "hand_raw": bool(hand_detected_raw),
                            "hand_auth": bool(hand_detected),
                            "faceid_enrolled": bool(self.face_id.enrolled),
                            "faceid_enrolling": bool(self.face_id.enrolling),
                            "faceid_progress": f"{n}/{N}",
                            "faceid_score": float(self.face_id.last_score),
                            "t_any": float(time_since_any),
                        })
                        self._prev_mode = mode
                        self._prev_cmd = command_str
                        self._prev_llm_reason = llm_reason
                        self._last_log_ts = now

                    # --- Overlay (clean HUD) ---
                    draw_hud(
                        frame,
                        mode=mode,
                        flying=self.flying,
                        hand_detected_raw=hand_detected_raw,
                        hand_detected=hand_detected,
                        raw_face=raw_face,
                        face_detected=face_detected,
                        gesture_name=gesture_name,
                        face_id=self.face_id,
                        llm_reason=llm_reason,
                    )

                    cv2.imshow("TELLO", frame)

                else:
                    # Match your actual stream size if you know it; otherwise keep 240x320
                    h, w = 240, 320
                    blank = 255 * (cv2.UMat(h, w, cv2.CV_8UC3).get())
                    cv2.putText(blank, "Waiting for video...", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.imshow("TELLO", blank)

                # --- Keyboard ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                # FaceID keys
                self._handle_faceid_keys(key)

                if key == ord("t"):
                    ok2, resp2 = self.tello.send_cmd("takeoff", timeout_ms=8000)
                    print("takeoff:", ok2, resp2)
                    self.flying = ok2 and (resp2 or "").lower() == "ok"

                if key == ord("l"):
                    ok2, resp2 = self.tello.send_cmd("land", timeout_ms=8000)
                    print("land:", ok2, resp2)
                    self.flying = False

                if key == ord("e"):
                    ok2, resp2 = self.tello.send_cmd("emergency", timeout_ms=3000)
                    print("emergency:", ok2, resp2)
                    self.flying = False

                # Manual RC override
                if self.flying and (not self._warned_deadband):
                    try:
                        if int(self.cfg.rc_deadband) >= int(self.cfg.rc_speed):
                            print(f"[WARN] rc_deadband ({self.cfg.rc_deadband}) >= rc_speed ({self.cfg.rc_speed}). "
                                  "Gesture/face RC may be clamped to 0. Lower deadband or raise speed.")
                        self._warned_deadband = True
                    except Exception:
                        self._warned_deadband = True

                if self.flying:
                    krc = rc_from_key(key, self.cfg.rc_speed)
                    if krc.active:
                        rc = krc

                # --- Send RC at fixed rate ---
                now2 = time.time()
                if now2 - last_rc_send >= self.cfg.rc_dt:
                    if self.flying:
                        self.tello.send_rc(
                            rc.lr, rc.fb, rc.ud, rc.yaw,
                            limit=self.cfg.rc_limit,
                            deadband=self.cfg.rc_deadband,
                        )
                    last_rc_send = now2

                self.logger.add(self.state.snapshot())

        finally:
            try:
                self.reasoner.stop()
            except Exception:
                pass
            try:
                if self.flying:
                    self.tello.send_cmd("land", timeout_ms=8000)
            except Exception:
                pass
            try:
                self.tello.send_cmd("streamoff", timeout_ms=2000)
            except Exception:
                pass

            self.video.stop()
            self.state.stop()
            self.tello.close()
            cv2.destroyAllWindows()

            try:
                self.logger.export()
                print(f"Telemetry saved to: {self.cfg.log_path}")
            except Exception as e:
                print("Telemetry export failed:", e)

            try:
                self.decisions.export()
                print(f"Decisions saved to: {self.decisions.path}")
            except Exception as e:
                print("Decision export failed:", e)

        return 0
