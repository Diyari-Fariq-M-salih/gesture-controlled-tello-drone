import cv2
import time
from dataclasses import dataclass
import mediapipe as mp
from .rc_command import RCCommand

def _clamp(v, lo, hi):
    return int(max(lo, min(hi, v)))

@dataclass
class FaceFollowConfig:
    target_area_frac: float = 0.07
    kp_yaw: float = 0.12
    kp_ud: float = 0.12
    kp_fb: float = 0.25
    max_yaw: int = 60
    max_ud: int = 50
    max_fb: int = 40
    deadband_px: int = 18
    deadband_area: float = 0.01
    lost_timeout_s: float = 0.7

class FaceFollower:
    def __init__(self, cfg: FaceFollowConfig | None = None):
        self.cfg = cfg or FaceFollowConfig()
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        self.last_face_time = 0.0

    def update(self, frame_bgr):
        """Returns (RCCommand, debug_frame_bgr)."""
        cfg = self.cfg
        frame = frame_bgr
        H, W = frame.shape[:2]
        frame_area = float(W * H)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)

        cmd = RCCommand(0, 0, 0, 0, active=False)

        if res.detections:
            # pick largest face
            best = None
            best_area = 0.0
            for det in res.detections:
                bbox = det.location_data.relative_bounding_box
                bw = bbox.width * W
                bh = bbox.height * H
                a = bw * bh
                if a > best_area:
                    best_area = a
                    best = det

            det = best
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * W)
            y = int(bbox.ymin * H)
            bw = int(bbox.width * W)
            bh = int(bbox.height * H)
            cx = x + bw // 2
            cy = y + bh // 2

            # draw debug
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            ex = cx - (W // 2)
            ey = cy - (H // 2)
            area_frac = (bw * bh) / frame_area
            
            # # HARD SAFETY LIMIT (never get closer than this)
            # MAX_AREA_FRAC = 0.06  # ~60 cm
            # if area_frac > MAX_AREA_FRAC:
            #     fb = -30  # force back away
            ez = cfg.target_area_frac - area_frac

            # proportional control
            yaw = _clamp(cfg.kp_yaw * ex, -cfg.max_yaw, cfg.max_yaw)
            ud  = _clamp(-cfg.kp_ud * ey, -cfg.max_ud, cfg.max_ud)
            fb  = _clamp(cfg.kp_fb * (ez * 1000.0), -cfg.max_fb, cfg.max_fb)

            # deadbands
            if abs(ex) < cfg.deadband_px: yaw = 0
            if abs(ey) < cfg.deadband_px: ud = 0
            if abs(ez) < cfg.deadband_area: fb = 0

            cmd = RCCommand(lr=0, fb=fb, ud=ud, yaw=yaw, active=True)
            self.last_face_time = time.time()

            cv2.putText(frame, f"Face ex:{ex} ey:{ey} area:{area_frac:.3f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # lost-face behavior handled by caller or here:
        if cmd.active and (time.time() - self.last_face_time) > cfg.lost_timeout_s:
            cmd = RCCommand(0, 0, 0, 0, active=True)  # hover

        return cmd, frame
