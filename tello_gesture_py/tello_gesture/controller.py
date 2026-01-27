import time
import cv2
from typing import Optional

from .config import ControllerConfig
from .tello_udp import TelloUDP
from .state_listener import StateListener
from .latest_frame import LatestFrame
from .video_stream import VideoStream
from .hand_gesture import HandGesture
from .gesture_logic import RuleBasedGesture
from .telemetry_logger import TelemetryLogger
from .keyboard import rc_from_key, RC

class Controller:
    def __init__(self, cfg: ControllerConfig, model_path: Optional[str] = None, labels_path: Optional[str] = None):
        self.cfg = cfg
        self.tello = TelloUDP(cfg.tello_ip, cfg.cmd_port, cfg.local_cmd_port)
        self.state = StateListener(cfg.state_port)
        self.latest = LatestFrame()
        self.video = VideoStream(self.latest, f"udp://0.0.0.0:{cfg.video_port}")
        self.hand = HandGesture(max_num_hands=1)

        self.rule = RuleBasedGesture(cfg.dir_thr, cfg.scale_thr, cfg.ema_alpha)

        self.keyboard_mode = False
        self.flying = False

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

    def run(self) -> int:
        if not self._sdk_init():
            return 1

        self.state.start()
        if not self.video.start():
            print("Video stream not opened. On Windows, check firewall for UDP 11111.")

        cv2.namedWindow("TELLO", cv2.WINDOW_NORMAL)

        rc = RC()
        last_rc_send = time.time()

        try:
            while True:
                ok, frame, seq, ts = self.latest.get(copy=True)

                if ok and frame is not None:
                    det = self.hand.detect(frame)

                    gesture_name = "NOHAND"
                    if det.has_hand and det.landmarks is not None:
                        if self._trained is not None:
                            gr = self._trained.predict(det.landmarks)
                        else:
                            gr = self.rule.predict(det.landmarks)

                        gesture_name = gr.name
                        self._last_hand_ts = time.time()

                        if not self.keyboard_mode and self.flying:
                            rc = self._rc_from_gesture(gesture_name, self.cfg.rc_speed)

                    else:
                        if (time.time() - self._last_hand_ts) > self.cfg.gesture_hold_s:
                            if not self.keyboard_mode:
                                rc = RC()

                    st = self.state.snapshot()
                    cv2.putText(frame, f"mode={'KEY' if self.keyboard_mode else 'GEST'} fly={'Y' if self.flying else 'N'}",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.putText(frame, f"gesture={gesture_name}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                    bat = st.get("bat", None)
                    h = st.get("h", None)
                    if bat is not None:
                        cv2.putText(frame, f"bat={bat:.0f}%", (10, 85),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    if h is not None:
                        cv2.putText(frame, f"h={h:.0f}cm", (10, 115),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                    cv2.imshow("TELLO", frame)
                else:
                    blank = 255 * (cv2.UMat(240, 320, cv2.CV_8UC3).get())
                    cv2.putText(blank, "Waiting for video...", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                    cv2.imshow("TELLO", blank)

                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if key == ord("q"):
                        break
                    if key == ord("m"):
                        self.keyboard_mode = not self.keyboard_mode
                        rc = RC()
                    if key == ord("t"):
                        ok, resp = self.tello.send_cmd("takeoff", timeout_ms=8000)
                        print("takeoff:", ok, resp)
                        self.flying = ok and resp.lower() == "ok"
                    if key == ord("l"):
                        ok, resp = self.tello.send_cmd("land", timeout_ms=8000)
                        print("land:", ok, resp)
                        self.flying = False
                        rc = RC()
                    if key == ord("e"):
                        ok, resp = self.tello.send_cmd("emergency", timeout_ms=3000)
                        print("emergency:", ok, resp)
                        self.flying = False
                        rc = RC()
                    if key == 32:  # space
                        rc = RC()

                    if self.keyboard_mode and self.flying:
                        krc = rc_from_key(key, self.cfg.rc_speed)
                        if krc != RC():
                            rc = krc

                now = time.time()
                if now - last_rc_send >= self.cfg.rc_dt:
                    if self.flying:
                        self.tello.send_rc(rc.lr, rc.fb, rc.ud, rc.yaw,
                                           limit=self.cfg.rc_limit,
                                           deadband=self.cfg.rc_deadband)
                    last_rc_send = now

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

    def _rc_from_gesture(self, gesture: str, speed: int) -> RC:
        parts = gesture.split("-")
        lr = fb = ud = yaw = 0
        if "LEFT" in parts: lr = -speed
        if "RIGHT" in parts: lr = speed
        if "UP" in parts: ud = speed
        if "DOWN" in parts: ud = -speed
        if "FORWARD" in parts: fb = speed
        if "BACK" in parts: fb = -speed
        return RC(lr=lr, fb=fb, ud=ud, yaw=yaw)
