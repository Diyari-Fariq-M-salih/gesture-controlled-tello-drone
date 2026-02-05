import time
import cv2
from typing import Optional

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

from .mode_manager import LLMModeManager, LLMConfig


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
        self.face.cfg.lost_timeout_s = 2.0

        self.llm = LLMModeManager(LLMConfig(
            decision_hz=1.0,
            timeout_s=4.0,
            mode_lock_s=0.4,
            enforce_perception_priority=True,  # IMPORTANT
        ))
        self.flying = False

        self.logger = TelemetryLogger(fields=["bat", "h", "yaw"], path=cfg.log_path)
        dec_path = cfg.log_path.replace(".csv", "_decisions.csv")
        self.decisions = DecisionLogger(path=dec_path)

        self._last_hand_ts = time.time()
        self._last_face_ts = time.time()
        self._last_any_seen_ts = time.time()

        self._search_yaw = 18

        # Balanced throttling
        self._hand_frame_i = 0
        self._hand_every_n = 2
        self._last_hand_det = None

        self._face_frame_i = 0
        self._face_every_n = 4

        # Light streaking
        self._hand_streak = 0
        self._face_streak = 0
        self._hand_streak_on = 2
        self._face_streak_on = 1

        self._prev_mode = self.llm.mode
        self._prev_reason = ""

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
                    hand_detected = self._hand_streak >= self._hand_streak_on
                    if hand_detected:
                        self._last_hand_ts = now
                        self._last_any_seen_ts = now

                    # --- Face observe (throttled) ---
                    self._face_frame_i += 1
                    if self._face_frame_i % self._face_every_n == 0:
                        self.face.observe(frame)

                    raw_face = self.face.face_detected()
                    self._face_streak = min(self._face_streak + 1, 10) if raw_face else 0
                    face_detected = self._face_streak >= self._face_streak_on
                    if face_detected:
                        self._last_face_ts = now
                        self._last_any_seen_ts = now

                    st = self.state.snapshot()
                    bat = st.get("bat", None)
                    alt = st.get("h", None)

                    time_since_hand = now - self._last_hand_ts
                    time_since_face = now - self._last_face_ts
                    time_since_any = now - self._last_any_seen_ts

                    state_for_llm = {
                        "mode": self.llm.mode,
                        "hand_detected": bool(hand_detected),
                        "face_detected": bool(face_detected),
                        "time_since_hand_s": float(time_since_hand),
                        "time_since_face_s": float(time_since_face),
                        "time_since_any_seen_s": float(time_since_any),
                        "battery": bat if bat is None else float(bat),
                        "altitude_cm": alt if alt is None else float(alt),
                        "flying": bool(self.flying),
                    }

                    # tick schedules LLM, but may also immediately force gesture/face
                    self.llm.tick(state_for_llm)
                    mode, reason = self.llm.get()

                    # Log when mode/reason changes
                    if mode != self._prev_mode or (reason and reason != self._prev_reason):
                        self.decisions.add({
                            "mode_before": self._prev_mode,
                            "mode_after": mode,
                            "reason": reason,
                            "hand": hand_detected,
                            "face": face_detected,
                            "battery": bat,
                            "altitude_cm": alt,
                            "t_any": time_since_any,
                        })
                        self._prev_mode = mode
                        self._prev_reason = reason

                    # --- Execute mode ---
                    if mode == "gesture":
                        if hand_detected and det is not None and det.landmarks is not None:
                            if self._trained is not None:
                                gr = self._trained.predict(det.landmarks)
                            else:
                                gr = self.rule.predict(det.landmarks)
                            gesture_name = gr.name
                            rc = rc_from_gesture_name(gesture_name, self.cfg.rc_speed) if self.flying else RC(active=False)
                        else:
                            gesture_name = "NOHAND"
                            rc = RC(active=True) if self.flying else RC(active=False)

                    elif mode == "face":
                        cmd, frame = self.face.update(frame)
                        rc = cmd if self.flying else RC(active=False)
                        gesture_name = "FACE"

                    elif mode == "search_360":
                        rc = RC(lr=0, fb=0, ud=0, yaw=self._search_yaw, active=True) if self.flying else RC(active=False)
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

                    # --- Overlay ---
                    cv2.putText(frame, f"MODE={mode.upper()} fly={'Y' if self.flying else 'N'}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"hand={int(hand_detected)} face={int(face_detected)} g={gesture_name}",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    if reason:
                        cv2.putText(frame, f"reason={reason[:70]}",
                                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                    frame_small = cv2.resize(frame, (640, 480))
                    cv2.imshow("TELLO", frame_small)

                else:
                    blank = 255 * (cv2.UMat(240, 320, cv2.CV_8UC3).get())
                    cv2.putText(blank, "Waiting for video...", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.imshow("TELLO", blank)

                # --- Keyboard ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

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
                if self.flying:
                    krc = rc_from_key(key, self.cfg.rc_speed)
                    if krc.active:
                        rc = krc

                # --- Send RC ---
                now2 = time.time()
                if now2 - last_rc_send >= self.cfg.rc_dt:
                    if self.flying:
                        self.tello.send_rc(
                            rc.lr, rc.fb, rc.ud, rc.yaw,
                            limit=self.cfg.rc_limit,
                            deadband=self.cfg.rc_deadband,
                        )
                    last_rc_send = now2

                # Telemetry
                self.logger.add(self.state.snapshot())

        finally:
            try:
                self.llm.stop()
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
