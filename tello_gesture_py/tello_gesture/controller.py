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
from .telemetry_logger import TelemetryLogger
from .keyboard import rc_from_key, RC

from .mode_manager import ModeManager, MODES
from .face_follow import FaceFollower
from face_recognition.face_recognizer_hybrid import FaceRecognizer

class Controller:
    def __init__(self, cfg: ControllerConfig, model_path: Optional[str] = None, labels_path: Optional[str] = None):
        self.cfg = cfg
        self.tello = TelloUDP(cfg.tello_ip, cfg.cmd_port, cfg.local_cmd_port)
        self.state = StateListener(cfg.state_port)
        self.latest = LatestFrame()
        self.video = VideoStream(self.latest, f"udp://0.0.0.0:{cfg.video_port}")
        self.hand = HandGesture(max_num_hands=2)  # Changed from 1 to 2 to detect both hands

        self.rule = RuleBasedGesture(cfg.dir_thr, cfg.scale_thr, cfg.ema_alpha)
        self.RED = (0, 0, 255) # BGR for OpenCV
        # modes: keyboard / gesture / face
        self.modes = ModeManager(mode=cfg.mode)  # keep your old default behavior (gesture) feel free to change
        print(f"Starting in mode: {self.modes.mode.upper()}")
        self.flying = False

        # face follower
        self.face = FaceFollower()

        # face recognizer (optional)
        # thresholds:
        # - det_thresh: higher -> fewer detections but more reliable; lower -> more detections but more false positives
        # - simil_thresh: higher -> stricter matching to the authorized face (fewer false positives but more false negatives); lower -> looser matching
        # typical values:
        # - during day/afternoon: det_thresh=0.75, simil_thresh=0.6
        # - at night: det_thresh=0.6, simil_thresh=0.45
        self.recognizer = FaceRecognizer(det_thresh=0.6, simil_thresh=0.45) if cfg.recognize_faces else None
    
        self.logger = TelemetryLogger(
            fields=["bat", "h", "tof", "yaw", "vgx", "vgy", "vgz"],
            path=cfg.log_path,
        )
        self._last_log = 0.0
        self._last_hand_ts = 0.0
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
        print(f"Switched mode: {self.modes.mode.upper()}")

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
                ok, frame, seq, ts = self.latest.get(copy=True)
                
                if ok and frame is not None:
                    # Initialize face follower dimensions on first valid frame
                    if self.face.img_shape is None:
                        self.face.img_shape = frame.shape[:2]
                    
                    # Initialize debug_frame to the original frame
                    debug_frame = frame
                    
                    # if face recognition enabled, run it first on the raw frame (before any annotations)
                    if recognition_enabled:
                        # Main recognition logic
                        auth_bbox = self.recognizer.recognize(frame)
                        # Get annotated image with all faces (authorized and intruders) marked for debug display
                        annotated = self.recognizer.get_annotated_image()
                        if annotated is not None:
                            debug_frame = annotated
                        # If no annotated image, keep using original frame

                    # --- Mode-specific perception & command generation ---
                    now = time.time()

                    # MODE 1: Hand gesture control
                    if self.modes.is_mode("gesture"):
                        # Case 1: Face recognition enabled -> use auth_bbox to select which hand to follow (if multiple detected)
                        if recognition_enabled and auth_bbox is not None:
                            det = self.hand.detect_auth(frame, auth_bbox)
                            # detect_auth already returns HandDetection with a list containing the selected hand
                            # Just extract the single hand array from the list
                            if det.has_hand and det.landmarks is not None and len(det.landmarks) > 0:
                                det.landmarks = det.landmarks[0]

                        # Case 2: No face recognition -> just use the first detected hand (if any)
                        else:
                            # get all detected hands as a list of HandDetection objects
                            det = self.hand.detect(frame) 
                            # keep only the first hand for compatibility with existing logic (and to avoid breaking the gesture classifier which expects one hand)
                            if det.has_hand and det.landmarks is not None and len(det.landmarks) > 0:
                                det.landmarks = det.landmarks[0]
                        
                        gesture_name = "NOHAND"

                        if det.has_hand and det.landmarks is not None and auth_bbox is not None:
                            if self._trained is not None:
                                gr = self._trained.predict(det.landmarks)
                            else:
                                gr = self.rule.predict(det.landmarks)

                            gesture_name = gr.name
                            self._last_hand_ts = now

                            if self.flying:
                                rc = rc_from_gesture_name(gesture_name, self.cfg.rc_speed)

                        else:
                            # if hand is gone for long enough -> hover (only in gesture mode)
                            if (now - self._last_hand_ts) > self.cfg.gesture_hold_s:
                                rc = RC(active=True)  # hover

                    # MODE 2: Face following
                    elif self.modes.is_mode("face"):
                        # Case 1: Face recognition enabled -> use auth_bbox of the authorized face for control
                        if recognition_enabled:
                            cmd, _ = self.face.update_from_bbox_dbg(auth_bbox, frame)
                            # print('cmd: left/right: ', cmd.lr, ' forward/back: ', cmd.fb, ' up/down: ', cmd.ud, ' yaw: ', cmd.yaw)
                        # Case 2: No face recognition -> just use the largest detected face
                        else:
                            cmd, debug_frame = self.face.update(frame)

                        if self.flying:
                            # face follower already returns active hover when appropriate
                            rc = cmd
                        else:
                            rc = RC(active=False)

                        gesture_name = "FACE"

                    else:
                        # keyboard mode: only changes RC on keypress (handled below)
                        gesture_name = "KEY"
                    # debug_frame.flags.writeable = True 
                    # --- Overlay ---
                    st = self.state.snapshot()
                    cv2.putText(
                        debug_frame,
                        f"mode={self.modes.mode.upper()} fly={'Y' if self.flying else 'N'}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        self.RED,
                        2,
                    )
                    cv2.putText(
                        debug_frame,
                        f"gesture={gesture_name}",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        self.RED,
                        2,
                    )

                    bat = st.get("bat", None)
                    h = st.get("h", None)
                    if bat is not None:
                        cv2.putText(
                            debug_frame,
                            f"bat={bat:.0f}%",
                            (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            self.RED,
                            2,
                        )
                    if h is not None:
                        cv2.putText(
                            debug_frame,
                            f"h={h:.0f}cm",
                            (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            self.RED,
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

                # --- Keyboard controls / mode switching ---
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if key == ord("q"):
                        break

                    # Mode switching:
                    if key == ord("1"):
                        self.modes.set_mode("keyboard")
                        rc = RC(active=True)  # hover
                    if key == ord("2"):
                        self.modes.set_mode("gesture")
                        rc = RC(active=True)  # hover
                    if key == ord("3"):
                        self.modes.set_mode("face")
                        rc = RC(active=True)  # hover
                    if key == ord("m"):
                        self._cycle_mode()
                        rc = RC(active=True)  # hover

                    if key == ord("t"):
                        ok, resp = self.tello.send_cmd("takeoff", timeout_ms=8000)
                        print("takeoff:", ok, resp)
                        self.flying = ok and resp.lower() == "ok"
                        rc = RC(active=True)  # hover after takeoff

                    if key == ord("l"):
                        ok, resp = self.tello.send_cmd("land", timeout_ms=8000)
                        print("land:", ok, resp)
                        self.flying = False
                        rc = RC(active=False)

                    if key == ord("e"):
                        ok, resp = self.tello.send_cmd("emergency", timeout_ms=3000)
                        print("emergency:", ok, resp)
                        self.flying = False
                        rc = RC(active=False)

                    if key == 32:  # space -> hover/stop
                        rc = RC(active=True)

                    # Keyboard control only applies when mode is keyboard
                    if self.modes.is_mode("keyboard") and self.flying:
                        krc = rc_from_key(key, self.cfg.rc_speed)
                        if krc.active:
                            rc = krc

                # --- Send RC at fixed rate (single sender -> no fighting) ---
                now = time.time()
                if now - last_rc_send >= self.cfg.rc_dt:
                    if self.flying:
                        # If no active command this tick, hover
                        if not getattr(rc, "active", False):
                            send_rc = RC(0, 0, 0, 0, active=True)
                        else:
                            send_rc = rc

                        self.tello.send_rc(
                            send_rc.lr,
                            send_rc.fb,
                            send_rc.ud,
                            send_rc.yaw,
                            limit=self.cfg.rc_limit,
                            deadband=self.cfg.rc_deadband,
                        )
                    last_rc_send = now

                # --- Telemetry logging ---
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

        return 0
