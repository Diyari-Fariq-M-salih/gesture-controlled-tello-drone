"""Follow the enrolled operator from behind when no face is visible.

The use case is carrying something for someone: the operator walks away and the
aircraft trails them at a set distance and height. Identity from behind comes
from the body, not the face:

  enrolment   After face enrolment, a countdown (default 6 s) lets the operator
              turn round; then ~20 crops of their back are embedded and averaged
              into a template. A crop is accepted only when no face is visible
              and exactly one person is in view, so the front, or a bystander,
              is not enrolled.
  matching    MediaPipe Pose finds people and crops each body; OSNet embeds
              each crop (person re-identification, K. Zhou et al., ICCV 2019;
              the x0_25 size, ~6 ms a crop through ONNX). Cosine similarity to
              the template, with on/off hysteresis like the face gate, picks the
              operator; a locked track is kept by overlap while it scores above
              the lower threshold.
  control     yaw keeps the operator centred; forward/back holds the set
              distance, estimated from the shoulder-to-hip length in pixels;
              up/down holds the set height from the aircraft's altimeter.

Appearance is mostly clothing, so this recognises an outfit, not a person:
someone dressed alike can match. Telling them apart is future work; the
match scores are logged per frame so it can be measured.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .rc_command import RCCommand
from .config import BackFollowConfig

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24


def _clamp(v, lo, hi):
    return int(max(lo, min(hi, v)))


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Person:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    torso_px: float
    crop: np.ndarray


@dataclass
class BackObs:
    detected: bool = False
    bbox: Optional[Tuple[int, int, int, int]] = None
    score: Optional[float] = None
    best_score: Optional[float] = None
    dist_m: Optional[float] = None
    n_people: int = 0
    ran_pose: bool = False
    state: str = "off"               # off | countdown | capturing | ready | following | lost


class BackReID:
    """OSNet through ONNX Runtime: a body crop to an L2-normalised 512-d embedding."""

    def __init__(self, model_path: str):
        import onnxruntime as ort
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"re-ID model not found at {model_path}; build it once with "
                "tello_gesture_py/scripts/export_osnet.py (see the README)")
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def embed(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), np.float32)
        x = np.stack([
            ((cv2.cvtColor(cv2.resize(c, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
              - _MEAN) / _STD).transpose(2, 0, 1) for c in crops])
        e = self.sess.run(None, {self.inp: x.astype(np.float32)})[0]
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)


class BackFollower:
    def __init__(self, cfg: Optional[BackFollowConfig] = None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.cfg = cfg or BackFollowConfig()
        self._mp = mp
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=Path(self.cfg.pose_model).read_bytes()),
            num_poses=int(self.cfg.num_poses), output_segmentation_masks=False))
        self.reid = BackReID(self.cfg.reid_model)
        self.template: Optional[np.ndarray] = None
        self.phase = "idle"              # idle | countdown | capturing
        self._t_phase = 0.0
        self._samples: List[np.ndarray] = []
        self._t_last_sample = 0.0
        self._people: List[Person] = []
        self._scores: List[float] = []
        self._lock_bbox = None
        self._last_match_t = 0.0
        self._n = 0
        self.last = BackObs()

    # ---------------------------------------------------------------- enrolment
    @property
    def enrolled(self) -> bool:
        return self.template is not None

    def start_enroll(self, now: float):
        self.phase, self._t_phase, self._samples = "countdown", now, []

    def cancel_enroll(self):
        self.phase, self._samples = "idle", []

    def clear(self):
        self.template, self._lock_bbox = None, None
        self.cancel_enroll()

    def enroll_status(self, now: float) -> str:
        if self.phase == "countdown":
            left = max(0.0, self.cfg.enroll_delay_s - (now - self._t_phase))
            return f"TURN AROUND  back enrol in {left:.0f}s"
        if self.phase == "capturing":
            return f"BACK ENROLLING {len(self._samples)}/{self.cfg.enroll_samples}"
        return ""

    # ---------------------------------------------------------------- perception
    def _detect(self, frame: np.ndarray) -> List[Person]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        out = []
        for lms in res.pose_landmarks or []:
            pts = np.array([(p.x * w, p.y * h, p.visibility if p.visibility is not None else 1.0)
                            for p in lms], np.float32)
            core = pts[[L_SH, R_SH, L_HIP, R_HIP]]
            shoulders_ok = (core[:2, 2] >= self.cfg.vis_thresh).all()
            hips_ok = (core[2:, 2] >= self.cfg.hip_vis_thresh).all()
            if not (shoulders_ok and hips_ok):
                continue                 # distance is judged from shoulders to hips
            sh, hp = core[:2, :2].mean(0), core[2:, :2].mean(0)
            torso = float(np.linalg.norm(sh - hp))
            if torso < self.cfg.min_torso_px:
                continue
            vis = pts[pts[:, 2] >= self.cfg.vis_thresh]
            x1, y1 = vis[:, 0].min(), vis[:, 1].min()
            x2, y2 = vis[:, 0].max(), vis[:, 1].max()
            px, py = 0.15 * (x2 - x1), 0.08 * (y2 - y1)
            b = (int(max(0, x1 - px)), int(max(0, y1 - py)),
                 int(min(w, x2 + px)), int(min(h, y2 + py)))
            if b[2] - b[0] < 16 or b[3] - b[1] < 32:
                continue
            out.append(Person(b, torso, frame[b[1]:b[3], b[0]:b[2]].copy()))
        return out

    def distance_m(self, torso_px: float, frame_w: int) -> float:
        f = self.cfg.focal_px_960 * frame_w / 960.0
        return f * self.cfg.torso_m / max(torso_px, 1e-6)

    def step(self, frame: np.ndarray, now: float, face_visible: bool) -> BackObs:
        """One frame. Pose runs only when there is something to do with it."""
        cfg = self.cfg
        if self.phase == "countdown" and now - self._t_phase >= cfg.enroll_delay_s:
            self.phase, self._t_phase = "capturing", now
        need = self.phase == "capturing" or (self.enrolled and not face_visible)
        obs = BackObs(state=self._state_name(now, face_visible))
        if not need:
            self._lock_bbox = None if face_visible else self._lock_bbox
            self.last = obs
            return obs

        self._n += 1
        if self._n % max(cfg.pose_every_n, 1) == 0 or not self._people:
            self._people = self._detect(frame)
            obs.ran_pose = True
            # embed only on fresh detections; between them the last result holds
            if self.phase == "capturing" or self.enrolled:
                emb = self.reid.embed([p.crop for p in self._people])
                self._scores = ([float(e @ self.template) for e in emb]
                                if self.enrolled else [])
                if self.phase == "capturing":
                    self._capture(emb, now, face_visible)
        obs.n_people = len(self._people)

        if self.enrolled and not face_visible and self.phase == "idle":
            self._match(obs, now, frame.shape[1])
        obs.state = self._state_name(now, face_visible)
        self.last = obs
        return obs

    def _capture(self, emb: np.ndarray, now: float, face_visible: bool):
        cfg = self.cfg
        if (len(self._people) == 1 and not face_visible
                and now - self._t_last_sample >= cfg.enroll_every_s):
            self._samples.append(emb[0])
            self._t_last_sample = now
        timed_out = now - self._t_phase > cfg.enroll_timeout_s
        if len(self._samples) >= cfg.enroll_samples or (
                timed_out and len(self._samples) >= cfg.enroll_min_samples):
            n = len(self._samples)
            t = np.mean(self._samples, axis=0)
            self.template = t / max(np.linalg.norm(t), 1e-12)
            self.phase, self._samples = "idle", []
            print(f"[back] enrolled from {n} crops")
        elif timed_out:
            print(f"[back] enrolment failed: {len(self._samples)} usable crops "
                  "(one person, back to the camera, whole torso in view)")
            self.cancel_enroll()

    def _match(self, obs: BackObs, now: float, frame_w: int):
        cfg = self.cfg
        if not self._people or not self._scores:
            if now - self._last_match_t > cfg.hold_s:
                self._lock_bbox = None
            obs.detected = now - self._last_match_t <= cfg.hold_s and self.last.bbox is not None
            obs.bbox, obs.dist_m = self.last.bbox, self.last.dist_m
            return
        obs.best_score = max(self._scores)
        pick = None
        if self._lock_bbox is not None:
            # a locked track stays locked while it overlaps and scores above `off`
            cand = [(i, _iou(p.bbox, self._lock_bbox)) for i, p in enumerate(self._people)]
            cand = [(i, o) for i, o in cand if o > 0.2 and self._scores[i] >= cfg.match_off]
            if cand:
                pick = max(cand, key=lambda c: (c[1], self._scores[c[0]]))[0]
        if pick is None:
            i = int(np.argmax(self._scores))
            if self._scores[i] >= cfg.match_on:
                pick = i
        if pick is None:
            if now - self._last_match_t > cfg.hold_s:
                self._lock_bbox = None
            return
        p = self._people[pick]
        self._lock_bbox, self._last_match_t = p.bbox, now
        dist = self.distance_m(p.torso_px, frame_w)
        # smooth the box and the distance so the command does not chatter; a fresh
        # lock after a loss starts from the new values
        prev = self.last if (self.last.detected and self.last.bbox is not None) else None
        k = cfg.smooth
        if prev is not None:
            box = tuple(int(round(k * n + (1 - k) * o)) for n, o in zip(p.bbox, prev.bbox))
            dist = k * dist + (1 - k) * (prev.dist_m if prev.dist_m is not None else dist)
        else:
            box = p.bbox
        obs.detected, obs.bbox, obs.score = True, box, self._scores[pick]
        obs.dist_m = dist

    def _state_name(self, now: float, face_visible: bool) -> str:
        if self.phase in ("countdown", "capturing"):
            return self.phase
        if not self.enrolled:
            return "off"
        if face_visible:
            return "ready"
        return "following" if now - self._last_match_t <= self.cfg.hold_s else "lost"

    # ---------------------------------------------------------------- control
    def command(self, obs: BackObs, frame_w: int, altitude_cm: Optional[float]) -> RCCommand:
        """Hold the operator centred, at the set distance and height."""
        cfg = self.cfg
        if not obs.detected or obs.bbox is None:
            return RCCommand(active=True)
        ex = (obs.bbox[0] + obs.bbox[2]) / 2.0 - frame_w / 2.0
        yaw = 0 if abs(ex) < cfg.deadband_px else _clamp(cfg.kp_yaw * ex, -cfg.max_yaw, cfg.max_yaw)
        fb = 0
        if obs.dist_m is not None:
            ed = obs.dist_m - cfg.follow_distance_m
            if abs(ed) >= cfg.deadband_m:
                fb = _clamp(cfg.kp_fb_per_m * ed, -cfg.max_fb, cfg.max_fb)
        ud = 0
        if altitude_cm is not None:
            eh = cfg.follow_height_m - float(altitude_cm) / 100.0
            if abs(eh) >= cfg.deadband_h_m:
                ud = _clamp(cfg.kp_ud_per_m * eh, -cfg.max_ud, cfg.max_ud)
        return RCCommand(lr=0, fb=fb, ud=ud, yaw=yaw, active=True)

    def adjust(self, key: int) -> Optional[str]:
        """[ ] follow distance, - = follow height."""
        c = self.cfg
        if key == ord("["):
            c.follow_distance_m = max(1.0, c.follow_distance_m - 0.25)
        elif key == ord("]"):
            c.follow_distance_m = min(6.0, c.follow_distance_m + 0.25)
        elif key == ord("-"):
            c.follow_height_m = max(0.5, round(c.follow_height_m - 0.1, 2))
        elif key == ord("="):
            c.follow_height_m = min(2.5, round(c.follow_height_m + 0.1, 2))
        else:
            return None
        return f"follow {c.follow_distance_m:.2f} m behind, {c.follow_height_m:.1f} m high"

    def draw(self, frame: np.ndarray, now: float):
        o = self.last
        msg = self.enroll_status(now)
        if msg:
            cv2.putText(frame, msg, (20, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, msg, (20, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255), 2, cv2.LINE_AA)
        if o.bbox is not None and o.detected:
            x1, y1, x2, y2 = o.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
            lab = f"BACK {o.score if o.score is not None else 0:.2f}"
            if o.dist_m is not None:
                lab += f"  {o.dist_m:.1f}/{self.cfg.follow_distance_m:.1f} m"
            cv2.putText(frame, lab, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 160, 0), 2, cv2.LINE_AA)

    def fields(self) -> dict:
        """Per-frame log columns."""
        o = self.last
        r = lambda v, n=3: None if v is None else round(float(v), n)
        return {"back_state": o.state, "back_detected": int(o.detected),
                "back_score": r(o.score), "back_best_score": r(o.best_score),
                "back_dist_m": r(o.dist_m, 2), "back_people": o.n_people,
                "back_ran_pose": int(o.ran_pose),
                "follow_distance_m": self.cfg.follow_distance_m,
                "follow_height_m": self.cfg.follow_height_m}
