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
from face_recognition.face_recognizer_hybrid import FaceRecognizer

from .mode_manager import (
    DeterministicModeManager,
    DeterministicConfig,
    LLMReasoner,
    LLMReasonConfig,
)


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


        # Face recognizer (Mediapipe + InsightFace)
        # thresholds:
        # - det_thresh: higher -> fewer detections but more reliable; lower -> more detections but more false positives
        # - simil_thresh: higher -> stricter matching to the authorized face (fewer false positives but more false negatives); lower -> looser matching
        # typical values:
        # - during day/afternoon: det_thresh=0.75, simil_thresh=0.6
        # - at night: det_thresh=0.6, simil_thresh=0.45
        self.recognizer = FaceRecognizer(det_thresh=0.65, simil_thresh=0.55) if cfg.recognize_faces else None
    
        # Deterministic mode manager
        self.mode_mgr = DeterministicModeManager(DeterministicConfig(
            battery_land_pct=15,
            nohuman_search_s=10.0,
            search_duration_s=5.0,     # quick spin window
            search_cooldown_s=10.0,    # prevent back-to-back spins
            mode_hold_s=1.2,
            hand_release_s=0.8,
            face_release_s=0.8,
        ))

        # Faster spin: raise yaw rate for ~5s "full-ish" sweep
        # Tello yaw command range is [-100..100]. 80 is aggressive but still within limits.
        self._search_yaw_cmd = 80

        # LLM reasoner (reason-only)
        self.reasoner = LLMReasoner(LLMReasonConfig(
            enabled=False,
            decision_hz=1.0,
            timeout_s=4.0,
        ))

        self.flying = False

        self.logger = TelemetryLogger(fields=["bat", "h", "yaw"], path=cfg.log_path)
        dec_path = cfg.log_path.replace(".csv", "_decisions.csv")
        self.decisions = DecisionLogger(path=dec_path)

        # Timers
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

        # Gesture stability gating (helps suppress 1-frame FORWARD/BACK spikes)
        self._gesture_stable_name = "CENTER"
        self._gesture_stable_count = 0
        self._gesture_fb_streak_on = 2  # require N consecutive frames for FORWARD/BACK

        # CSV logging control (log on changes + periodic)
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

        self.TEXT_COLOR = (0, 255, 0)  # Green text for better visibility

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

    # def _cycle_mode(self):
    #     cur = self.modes.mode
    #     idx = MODES.index(cur)
    #     self.modes.set_mode(MODES[(idx + 1) % len(MODES)])
    #     print(f"Switched mode: {self.modes.mode.upper()}")

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

        # flag to indicate if face recognition is enabled (used multiple times in the loop)
        recognition_enabled = self.cfg.recognize_faces and self.recognizer is not None
        auth_bbox = None
        # init frame dimensions for face follower (will be captured from first frame)
        ok, frame, _, _ = self.latest.get(copy=True)
        if ok and frame is not None:
            self.face.img_shape = frame.shape[:2]
        else:
            print("Error reading first frame, using default size for face follower.")
            self.face.img_shape = [720, 960] # H, W
        
        ## MAIN LOOP ##
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
                    # if recognition enabled, run recognizer every N frames to keep it from being a bottleneck. If not enabled, run regular face follower detect.
                    if recognition_enabled:
                        if self._face_frame_i % self._face_every_n == 0:
                            auth_bbox = self.recognizer.recognize(frame)
                        raw_face = self.recognizer.face_detected()
                    else:
                        if self._face_frame_i % self._face_every_n == 0:
                            self.face.observe(frame)
                        raw_face = self.face.face_detected()
                        
                    self._face_streak = min(self._face_streak + 1, 10) if raw_face else 0
                    # face is considered detected if it has been observed for N consecutive checks (helps with jitter on low FPS streams)
                    face_detected = self._face_streak >= self._face_streak_on
                    if face_detected:
                        self._last_face_ts = now
                        self._last_any_seen_ts = now

                    # --- Telemetry snapshot ---
                    st = self.state.snapshot()
                    bat = st.get("bat", None)
                    alt = st.get("h", None)

                    time_since_hand = now - self._last_hand_ts
                    time_since_face = now - self._last_face_ts
                    time_since_any = now - self._last_any_seen_ts

                    # If the hand has been gone long enough, reset temporal gesture state.
                    # This prevents stale scale history from causing random FORWARD/BACK when the hand reappears.
                    if (now - self._last_hand_ts) > float(self.mode_mgr.cfg.hand_release_s):
                        try:
                            self.rule.reset()
                        except Exception:
                            pass
                    # current detection state for mode manager (used for deterministic mode decisions and LLM reasoning)
                    state_for_mode = {
                        "recognition_enabled": bool (recognition_enabled),
                        "hand_detected": bool(hand_detected),
                        "face_detected": bool(face_detected),
                        "time_since_hand_s": float(time_since_hand),
                        "time_since_face_s": float(time_since_face),
                        "time_since_any_seen_s": float(time_since_any),
                        "battery": bat if bat is None else float(bat),
                        "altitude_cm": alt if alt is None else float(alt),
                        "flying": bool(self.flying),
                    }

                    # for debugging:
                    # print(f"State: Hand detected: {hand_detected}, Face detected: {face_detected}, \n")
                    # print(f"Time since hand: {time_since_hand:.1f}s, Time since face: {time_since_face:.1f}s")
                    
                    mode, det_reason = self.mode_mgr.tick(state_for_mode)

                    # --- Execute deterministic mode ---
                    # MODE 1: Hand gesture control
                    if mode == "gesture":
                        # Case 1: Face recognition enabled -> use auth_bbox to select which hand to follow (if multiple detected)
                        if recognition_enabled and auth_bbox is not None:
                            det = self.hand.detect_auth(frame, auth_bbox)
                            # Just extract the single hand array from the list
                            if det.has_hand and det.landmarks is not None and len(det.landmarks) > 0:
                                det.landmarks = det.landmarks[0]

                        # Case 2: No face recognition -> just use the most recently detected hand (if any)
                        else:
                            # get all detected hands as a list of HandDetection objects
                            # det = self.hand.detect(frame)
                            # det is a list of HandDetection objects
                            # But we only keep the first hand landmarks to avoid breaking the gesture classifier which expects one hand
                            # this was previously done inside the classifier, but doing it here allows us to still get multiple hand detections for the recognition case
                            if det.has_hand and det.landmarks is not None and len(det.landmarks) > 0:
                                det.landmarks = det.landmarks[0]

                        if hand_detected and det is not None and det.landmarks is not None:
                            if self._trained is not None:
                                gr = self._trained.predict(det.landmarks)
                            else:
                                gr = self.rule.predict(det.landmarks)

                            # Track stability of the predicted gesture label
                            if gr.name == self._gesture_stable_name:
                                self._gesture_stable_count = min(self._gesture_stable_count + 1, 50)
                            else:
                                self._gesture_stable_name = gr.name
                                self._gesture_stable_count = 1

                            gesture_name = gr.name

                            # FORWARD/BACK gating: require N consecutive frames before applying fb motion
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

                    # MODE 2: Face following
                    elif mode == "face":
                        # Case 1: Face recognition enabled -> use auth_bbox of the authorized face for control
                        if recognition_enabled:
                            cmd, frame = self.face.update_from_bbox_dbg(auth_bbox, frame)
                            # print('cmd: left/right: ', cmd.lr, ' forward/back: ', cmd.fb, ' up/down: ', cmd.ud, ' yaw: ', cmd.yaw)
                        # Case 2: No face recognition -> just use the largest detected face
                        else:
                            cmd, frame = self.face.update(frame)
                        # update RC command only if we're currently flying, otherwise keep it inactive to prevent drift when we take off
                        rc = cmd if self.flying else RC(active=False)
                        gesture_name = "FACE"

                    # MODE 3: 360 Search
                    elif mode == "search_360":
                        # fast spin command during the 5s search window
                        rc = RC(lr=0, fb=0, ud=0, yaw=self._search_yaw_cmd, active=True) if self.flying else RC(active=False)
                        gesture_name = "SEARCH"

                    # MODE 4: Hover (no control input, just maintain hover)
                    elif mode == "hover":
                        rc = RC(active=True) if self.flying else RC(active=False)
                        gesture_name = "HOVER"

                    # MODE 5: Land (immediate land command, overrides any control input)
                    elif mode == "land":
                        if self.flying:
                            self.tello.send_cmd("land", timeout_ms=8000)
                        self.flying = False
                        rc = RC(active=False)
                        gesture_name = "LAND"

                    # MODE 6: Other (default fallback)
                    else:
                        rc = RC(active=True) if self.flying else RC(active=False)
                        gesture_name = "OTHER"

                    command_str = self._rc_to_command_str(rc)

                    # --- LLM reason (reason-only, non-blocking) ---
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

                    # --- Log CSV: command + LLM reason (to compare if they match) ---
                    should_log = False
                    if (now - self._last_log_ts) >= self._log_every_s:
                        should_log = True
                    if mode != self._prev_mode or command_str != self._prev_cmd:
                        should_log = True
                    if llm_reason and llm_reason != self._prev_llm_reason:
                        should_log = True

                    if should_log:
                        self.decisions.add({
                            "mode": mode,
                            "command": command_str,
                            "llm_reason": llm_reason,
                            "det_reason": det_reason,
                            "battery": bat,
                            "altitude_cm": alt,
                            "face": bool(face_detected),
                            "hand": bool(hand_detected),
                            "t_any": float(time_since_any),
                        })
                        self._prev_mode = mode
                        self._prev_cmd = command_str
                        self._prev_llm_reason = llm_reason
                        self._last_log_ts = now

                    # --- Overlay ---
                    cv2.putText(frame, f"MODE={mode.upper()} fly={'Y' if self.flying else 'N'}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 2)
                    cv2.putText(frame, f"hand={int(hand_detected)} face={int(face_detected)} g={gesture_name}",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 2)
                    if llm_reason:
                        cv2.putText(frame, f"LLM: {llm_reason[:70]}",
                                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.TEXT_COLOR, 2)

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

                # Telemetry
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