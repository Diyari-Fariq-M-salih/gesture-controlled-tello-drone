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
from .telemetry_logger import TelemetryLogger, DecisionLogger
from .keyboard import rc_from_key, RC

from .mode_manager import ModeManager, MODES, LLMModeChooser, LLMConfig
from .face_follow import FaceFollower


class Controller:
    def __init__(self, cfg: ControllerConfig, model_path: Optional[str] = None, labels_path: Optional[str] = None):
        self.cfg = cfg
        self.tello = TelloUDP(cfg.tello_ip, cfg.cmd_port, cfg.local_cmd_port)
        self.state = StateListener(cfg.state_port)
        self.latest = LatestFrame()

        # Keep your working URL/port (firewall now allows it)
        self.video = VideoStream(self.latest, f"udp://0.0.0.0:{cfg.video_port}?fifo_size=5000000&overrun_nonfatal=1")

        self.hand = HandGesture(max_num_hands=1)
        self.rule = RuleBasedGesture(cfg.dir_thr, cfg.scale_thr, cfg.ema_alpha)

        llm_cfg = LLMConfig(
            enabled=True,
            model="qwen2.5:0.5b-instruct",
            decision_hz=0.5,
            timeout_s=3.5,
        )
        self.modes = ModeManager(mode="hover", llm=LLMModeChooser(llm_cfg))
        self.flying = False

        self.face = FaceFollower()

        self.logger = TelemetryLogger(
            fields=["bat", "h", "tof", "yaw", "vgx", "vgy", "vgz"],
            path=cfg.log_path,
        )
        dec_path = cfg.log_path
        if dec_path.lower().endswith(".csv"):
            dec_path = dec_path[:-4] + "_decisions.csv"
        else:
            dec_path = dec_path + "_decisions.csv"
        self.decision_logger = DecisionLogger(path=dec_path)

        self._last_log = 0.0
        self._last_hand_ts = time.time()
        self._last_key_ts = 0.0
        self._last_key = ""

        # SEARCH_360 params (gentle yaw)
        self._search_start_ts: float = 0.0
        self._search_duration_s: float = 10.0
        self._search_yaw: int = 18

        # Hand detection throttling
        self._hand_frame_i = 0
        self._hand_every_n = 2
        self._last_hand_det = None

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

    def _cycle_mode(self):
        cur = self.modes.mode
        idx = MODES.index(cur)
        self.modes.set_mode(MODES[(idx + 1) % len(MODES)])

    def run(self) -> int:
        if not self._sdk_init():
            return 1

        self.state.start()
        if not self.video.start():
            print("Video stream not opened. On Windows, check firewall for UDP 11111.")

        cv2.namedWindow("TELLO", cv2.WINDOW_NORMAL)

        rc = RC(active=False)
        last_rc_send = time.time()
        gesture_name = "NOHAND"
        last_reason = ""

        try:
            while True:
                ok, frame, seq, ts = self.latest.get(copy=True)

                if ok and frame is not None:
                    debug_frame = frame
                    now = time.time()

                    # Key "decays" so it doesn't stick forever in LLM input
                    if (now - self._last_key_ts) > 1.0:
                        self._last_key = ""

                    # Throttled hand detection
                    self._hand_frame_i += 1
                    det = None
                    if self._hand_frame_i % self._hand_every_n == 0:
                        det = self.hand.detect(frame)
                        self._last_hand_det = det
                    else:
                        det = self._last_hand_det

                    hand_detected = bool(det is not None and det.has_hand and det.landmarks is not None)

                    # Gesture classification
                    gesture_conf = 0.0
                    if hand_detected:
                        if self._trained is not None:
                            gr = self._trained.predict(det.landmarks)
                        else:
                            gr = self.rule.predict(det.landmarks)
                        gesture_name = gr.name
                        gesture_conf = float(getattr(gr, "confidence", 0.0))
                        self._last_hand_ts = now
                    else:
                        gesture_name = "NOHAND"

                    time_since_hand_s = (now - self._last_hand_ts) if self._last_hand_ts > 0 else 1e9

                    # Keep face signals fresh when NOT in face mode
                    if not self.modes.is_mode("face"):
                        self.face.observe(frame)

                    face_detected = self.face.face_detected()
                    time_since_face_s = self.face.time_since_face_s()

                    # Telemetry
                    st = self.state.snapshot()
                    bat = st.get("bat", None)
                    h = st.get("h", None)

                    # Deterministic: in FACE mode, if face lost >= 10s -> SEARCH_360
                    if self.flying and self.modes.is_mode("face") and time_since_face_s >= 10.0:
                        mb = self.modes.mode
                        self.modes.set_mode("search_360")
                        self._search_start_ts = 0.0
                        last_reason = "Hard rule: face lost >= 10s -> SEARCH_360"
                        self.decision_logger.add({
                            "mode_before": mb,
                            "mode_after": self.modes.mode,
                            "reason": last_reason,
                            "face_detected": face_detected,
                            "time_since_face_s": time_since_face_s,
                            "hand_detected": hand_detected,
                            "time_since_hand_s": time_since_hand_s,
                            "gesture": gesture_name,
                            "gesture_conf": gesture_conf,
                            "key": self._last_key,
                            "battery": bat,
                            "altitude_cm": h,
                        })

                    # Deterministic: if hovering and we see signals, switch immediately (no LLM needed)
                    if self.flying and self.modes.is_mode("hover"):
                        if hand_detected:
                            mb = "hover"
                            self.modes.set_mode("gesture")
                            last_reason = "Hard rule: hand detected -> GESTURE"
                            self.decision_logger.add({"mode_before": mb, "mode_after": "gesture", "reason": last_reason})
                        elif face_detected:
                            mb = "hover"
                            self.modes.set_mode("face")
                            last_reason = "Hard rule: face detected -> FACE"
                            self.decision_logger.add({"mode_before": mb, "mode_after": "face", "reason": last_reason})


                    # LLM decision tick (slow; log every tick that returns a reason)
                    mode_before = self.modes.mode
                    if self.flying:
                        state_for_llm = {
                            "mode": self.modes.mode,
                            "face_detected": face_detected,
                            "time_since_face_s": float(time_since_face_s),
                            "hand_detected": hand_detected,
                            "time_since_hand_s": float(time_since_hand_s),
                            "gesture": gesture_name,
                            "gesture_conf": float(gesture_conf),
                            "key": self._last_key,
                            "battery": bat,
                            "altitude_cm": h,
                            "flying": True,
                        }
                        mode_after, reason = self.modes.maybe_update_from_llm(state_for_llm)
                        if reason:
                            last_reason = reason
                            self.decision_logger.add({
                                "mode_before": mode_before,
                                "mode_after": mode_after,
                                "reason": reason,
                                "face_detected": face_detected,
                                "time_since_face_s": time_since_face_s,
                                "hand_detected": hand_detected,
                                "time_since_hand_s": time_since_hand_s,
                                "gesture": gesture_name,
                                "gesture_conf": gesture_conf,
                                "key": self._last_key,
                                "battery": bat,
                                "altitude_cm": h,
                            })

                    # Execute current mode (deterministic RC)
                    if self.modes.is_mode("gesture"):
                        if hand_detected and self.flying:
                            rc = rc_from_gesture_name(gesture_name, self.cfg.rc_speed)
                        else:
                            if time_since_hand_s > self.cfg.gesture_hold_s:
                                rc = RC(active=True)  # hover

                    elif self.modes.is_mode("face"):
                        cmd, debug_frame = self.face.update(frame)
                        rc = cmd if self.flying else RC(active=False)
                        gesture_name = "FACE"

                    elif self.modes.is_mode("search_360"):
                        if self._search_start_ts <= 0:
                            self._search_start_ts = now

                        # If face reacquired, go back to face mode
                        if face_detected:
                            self.modes.set_mode("face")
                            self._search_start_ts = 0.0
                            rc = RC(active=True)
                        else:
                            elapsed = now - self._search_start_ts
                            if elapsed >= self._search_duration_s:
                                self.modes.set_mode("hover")
                                self._search_start_ts = 0.0
                                rc = RC(active=True)
                            else:
                                rc = RC(lr=0, fb=0, ud=0, yaw=self._search_yaw, active=True)

                        gesture_name = "SEARCH"

                    elif self.modes.is_mode("hover"):
                        rc = RC(active=True)
                        gesture_name = "HOVER"

                    elif self.modes.is_mode("land"):
                        if self.flying:
                            ok2, resp2 = self.tello.send_cmd("land", timeout_ms=8000)
                            print("land:", ok2, resp2)
                        self.flying = False
                        self.modes.set_mode("hover")
                        rc = RC(active=False)
                        gesture_name = "LAND"

                    else:
                        gesture_name = "KEY"

                    # Overlay
                    cv2.putText(
                        debug_frame,
                        f"mode={self.modes.mode.upper()} fly={'Y' if self.flying else 'N'}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        debug_frame,
                        f"gesture={gesture_name}",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    if bat is not None:
                        cv2.putText(
                            debug_frame,
                            f"bat={bat:.0f}%",
                            (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                    if h is not None:
                        cv2.putText(
                            debug_frame,
                            f"h={h:.0f}cm",
                            (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                        )
                    if last_reason:
                        cv2.putText(
                            debug_frame,
                            f"reason={last_reason[:70]}",
                            (10, 175),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                        )

                    cv2.imshow("TELLO", debug_frame)

                else:
                    blank = 255 * (cv2.UMat(240, 320, cv2.CV_8UC3).get())
                    cv2.putText(
                        blank,
                        "Waiting for video...",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2,
                    )
                    cv2.imshow("TELLO", blank)

                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self._last_key_ts = time.time()
                    self._last_key = chr(key) if 0 <= key <= 255 else ""

                    if key == ord("q"):
                        break

                    if key == ord("1"):
                        self.modes.set_mode("keyboard")
                        rc = RC(active=True)
                    if key == ord("2"):
                        self.modes.set_mode("gesture")
                        rc = RC(active=True)
                    if key == ord("3"):
                        self.modes.set_mode("face")
                        rc = RC(active=True)
                    if key == ord("m"):
                        self._cycle_mode()
                        rc = RC(active=True)

                    if key == ord("t"):
                        ok2, resp2 = self.tello.send_cmd("takeoff", timeout_ms=8000)
                        print("takeoff:", ok2, resp2)
                        self.flying = ok2 and resp2.lower() == "ok"
                        rc = RC(active=True)

                    if key == ord("l"):
                        ok2, resp2 = self.tello.send_cmd("land", timeout_ms=8000)
                        print("land:", ok2, resp2)
                        self.flying = False
                        rc = RC(active=False)

                    if key == ord("e"):
                        ok2, resp2 = self.tello.send_cmd("emergency", timeout_ms=3000)
                        print("emergency:", ok2, resp2)
                        self.flying = False
                        rc = RC(active=False)

                    if key == 32:
                        rc = RC(active=True)

                    if self.modes.is_mode("keyboard") and self.flying:
                        krc = rc_from_key(key, self.cfg.rc_speed)
                        if krc.active:
                            rc = krc

                # Send RC at fixed rate
                now = time.time()
                if now - last_rc_send >= self.cfg.rc_dt:
                    if self.flying:
                        send_rc = RC(0, 0, 0, 0, active=True) if not getattr(rc, "active", False) else rc
                        self.tello.send_rc(
                            send_rc.lr,
                            send_rc.fb,
                            send_rc.ud,
                            send_rc.yaw,
                            limit=self.cfg.rc_limit,
                            deadband=self.cfg.rc_deadband,
                        )
                    last_rc_send = now

                # Telemetry logging
                if now - self._last_log >= (1.0 / self.cfg.log_hz):
                    self.logger.add(self.state.snapshot())
                    self._last_log = now

        finally:
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
                self.decision_logger.export()
                print(f"Decisions saved to: {self.decision_logger.path}")
            except Exception as e:
                print("Decision export failed:", e)

        return 0
