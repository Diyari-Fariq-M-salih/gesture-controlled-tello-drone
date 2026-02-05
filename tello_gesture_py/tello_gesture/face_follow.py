import cv2
import time
from dataclasses import dataclass
import mediapipe as mp
from .rc_command import RCCommand


def _clamp(v, lo, hi):
    return int(max(lo, min(hi, v)))


@dataclass
class FaceFollowConfig:
    target_area_frac: float = 0.075

    kp_yaw: float = 0.12
    kp_ud: float = 0.12
    kp_fb: float = 0.30

    max_yaw: int = 55
    max_ud: int = 40
    max_fb: int = 40

    deadband_px: int = 18
    deadband_area: float = 0.008

    lost_timeout_s: float = 0.7

    # Cheaper detection -> smoother stream
    detect_w: int = 256
    detect_h: int = 192
    detect_every_n: int = 5
    control_hz: float = 15.0

    area_ema_alpha: float = 0.25


class FaceFollower:
    def __init__(self, cfg: FaceFollowConfig | None = None):
        self.cfg = cfg or FaceFollowConfig()

        mp_face = mp.solutions.face_detection
        self.detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

        self._frame_count = 0
        self._last_face_time = 0.0

        self._last_cmd = RCCommand(0, 0, 0, 0, active=True)
        self._last_bbox = None
        self._last_area_frac = None
        self._area_ema = None

        self._last_control_ts = 0.0

    def face_detected(self) -> bool:
        return (time.time() - self._last_face_time) <= self.cfg.lost_timeout_s

    def time_since_face_s(self) -> float:
        if self._last_face_time <= 0:
            return 1e9
        return max(0.0, time.time() - self._last_face_time)

    def observe(self, frame_bgr):
        """Update face detection state WITHOUT generating RC commands."""
        cfg = self.cfg
        self._frame_count += 1
        now = time.time()

        H, W = frame_bgr.shape[:2]

        run_det = (self._frame_count % max(cfg.detect_every_n, 1) == 0) or (self._last_bbox is None)
        if not run_det:
            return

        small = cv2.resize(frame_bgr, (cfg.detect_w, cfg.detect_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res = self.detector.process(rgb)

        if not res.detections:
            return

        best = None
        best_area = 0.0
        for det in res.detections:
            bbox = det.location_data.relative_bounding_box
            bw = bbox.width * cfg.detect_w
            bh = bbox.height * cfg.detect_h
            a = bw * bh
            if a > best_area:
                best_area = a
                best = det

        bbox = best.location_data.relative_bounding_box
        sx = int(bbox.xmin * cfg.detect_w)
        sy = int(bbox.ymin * cfg.detect_h)
        sw = int(bbox.width * cfg.detect_w)
        sh = int(bbox.height * cfg.detect_h)

        scale_x = W / float(cfg.detect_w)
        scale_y = H / float(cfg.detect_h)
        x = int(sx * scale_x)
        y = int(sy * scale_y)
        w = int(sw * scale_x)
        h = int(sh * scale_y)

        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))
        w = max(1, min(W - x, w))
        h = max(1, min(H - y, h))

        self._last_bbox = (x, y, w, h)
        self._last_face_time = now

        area_frac = (w * h) / float(W * H)
        if self._area_ema is None:
            self._area_ema = area_frac
        else:
            a = cfg.area_ema_alpha
            self._area_ema = a * area_frac + (1.0 - a) * self._area_ema
        self._last_area_frac = self._area_ema

    def update(self, frame_bgr):
        """Returns (RCCommand, debug_frame_bgr)"""
        cfg = self.cfg
        self._frame_count += 1
        now = time.time()

        min_dt = 1.0 / max(cfg.control_hz, 1e-6)
        if (now - self._last_control_ts) < min_dt:
            dbg = frame_bgr
            self._draw_overlay(dbg)
            return self._last_cmd, dbg
        self._last_control_ts = now

        H, W = frame_bgr.shape[:2]
        dbg = frame_bgr

        run_det = (self._frame_count % max(cfg.detect_every_n, 1) == 0) or (self._last_bbox is None)

        if run_det:
            small = cv2.resize(frame_bgr, (cfg.detect_w, cfg.detect_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res = self.detector.process(rgb)

            if res.detections:
                best = None
                best_area = 0.0
                for det in res.detections:
                    bbox = det.location_data.relative_bounding_box
                    bw = bbox.width * cfg.detect_w
                    bh = bbox.height * cfg.detect_h
                    a = bw * bh
                    if a > best_area:
                        best_area = a
                        best = det

                bbox = best.location_data.relative_bounding_box
                sx = int(bbox.xmin * cfg.detect_w)
                sy = int(bbox.ymin * cfg.detect_h)
                sw = int(bbox.width * cfg.detect_w)
                sh = int(bbox.height * cfg.detect_h)

                scale_x = W / float(cfg.detect_w)
                scale_y = H / float(cfg.detect_h)
                x = int(sx * scale_x)
                y = int(sy * scale_y)
                w = int(sw * scale_x)
                h = int(sh * scale_y)

                x = max(0, min(W - 1, x))
                y = max(0, min(H - 1, y))
                w = max(1, min(W - x, w))
                h = max(1, min(H - y, h))

                self._last_bbox = (x, y, w, h)
                self._last_face_time = now

                area_frac = (w * h) / float(W * H)
                if self._area_ema is None:
                    self._area_ema = area_frac
                else:
                    a = cfg.area_ema_alpha
                    self._area_ema = a * area_frac + (1.0 - a) * self._area_ema
                self._last_area_frac = self._area_ema

        cmd = RCCommand(0, 0, 0, 0, active=True)

        if self._last_bbox is not None and (now - self._last_face_time) <= cfg.lost_timeout_s:
            x, y, w, h = self._last_bbox
            cx = x + w // 2
            cy = y + h // 2

            ex = cx - (W // 2)
            ey = cy - (H // 2)

            area_frac = self._last_area_frac if self._last_area_frac is not None else (w * h) / float(W * H)
            ez = cfg.target_area_frac - area_frac

            yaw = _clamp(cfg.kp_yaw * ex, -cfg.max_yaw, cfg.max_yaw)
            ud = _clamp(-cfg.kp_ud * ey, -cfg.max_ud, cfg.max_ud)
            fb = _clamp(cfg.kp_fb * (ez * 1000.0), -cfg.max_fb, cfg.max_fb)

            if abs(ex) < cfg.deadband_px:
                yaw = 0
            if abs(ey) < cfg.deadband_px:
                ud = 0
            if abs(ez) < cfg.deadband_area:
                fb = 0

            cmd = RCCommand(lr=0, fb=fb, ud=ud, yaw=yaw, active=True)
        else:
            cmd = RCCommand(0, 0, 0, 0, active=True)

        self._last_cmd = cmd
        self._draw_overlay(dbg)
        return cmd, dbg

    def _draw_overlay(self, frame_bgr):
        cfg = self.cfg

        if self._last_bbox is not None:
            x, y, w, h = self._last_bbox
            if (time.time() - self._last_face_time) <= cfg.lost_timeout_s:
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2, lineType=cv2.LINE_AA)
                cx = x + w // 2
                cy = y + h // 2
                cv2.circle(frame_bgr, (cx, cy), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        area = self._last_area_frac if self._last_area_frac is not None else 0.0
        cv2.putText(
            frame_bgr,
            f"FaceFollow area:{area:.3f} target:{cfg.target_area_frac:.3f}",
            (10, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
