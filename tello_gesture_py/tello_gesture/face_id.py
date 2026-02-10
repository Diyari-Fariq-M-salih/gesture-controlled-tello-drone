from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os

import cv2
import numpy as np


def _l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps:
        return x
    return x / n


@dataclass
class FaceIDConfig:
    model_path: str = "models/arcface.onnx"
    input_size: int = 112

    # Many TF-exported ArcFace models expect RGB. If your scores look wrong, flip this.
    rgb: bool = True

    # Common ArcFace normalization: (x - 127.5) / 128
    mean: float = 127.5
    std: float = 128.0

    enroll_samples: int = 20
    enroll_min_face_px: int = 60

    cosine_thr: float = 0.55


class FaceID:
    """
    Identity lock using pretrained embedding model (ONNX) + enrollment template + cosine similarity.
    Uses onnxruntime (works with NHWC models that OpenCV DNN often fails on).
    """

    def __init__(self, cfg: Optional[FaceIDConfig] = None):
        self.cfg = cfg or FaceIDConfig()

        self.enabled: bool = False

        self.enrolled: bool = False
        self.enrolling: bool = False

        self._template: Optional[np.ndarray] = None
        self._samples: list[np.ndarray] = []

        self.last_score: float = -1.0

        self._sess = None
        self._in_name: Optional[str] = None
        self._layout: str = "NCHW"  # or NHWC

        self._load_onnx()

    def _load_onnx(self) -> None:
        mp = self.cfg.model_path
        if not mp or not os.path.exists(mp):
            print(f"[FaceID] Model not found: '{mp}' (FaceID disabled)")
            self.enabled = False
            return

        try:
            import onnxruntime as ort

            self._sess = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
            self._in_name = self._sess.get_inputs()[0].name
            ishape = self._sess.get_inputs()[0].shape  # can include None

            layout = "NCHW"
            if isinstance(ishape, (list, tuple)) and len(ishape) == 4:
                # Prefer explicit inference from known dims
                if ishape[-1] == 3:
                    layout = "NHWC"
                elif ishape[1] == 3:
                    layout = "NCHW"
            self._layout = layout

            self.enabled = True
            print(f"[FaceID] Loaded embedding model: {mp} (layout={self._layout})")
        except Exception as e:
            print(f"[FaceID] Failed to load ONNX with onnxruntime: {type(e).__name__}: {e}")
            self.enabled = False
            self._sess = None
            self._in_name = None

    def start_enroll(self) -> None:
        if not self.enabled:
            self._load_onnx()
        self.enrolling = True
        self._samples = []
        self.last_score = -1.0

    def cancel_enroll(self) -> None:
        self.enrolling = False
        self._samples = []

    def clear(self) -> None:
        self.enrolled = False
        self._template = None
        self.enrolling = False
        self._samples = []
        self.last_score = -1.0

    def enroll_progress(self) -> Tuple[int, int]:
        return len(self._samples), int(self.cfg.enroll_samples)

    def add_sample(self, face_bgr: np.ndarray) -> None:
        if not self.enrolling:
            return
        if face_bgr is None:
            return
        h, w = face_bgr.shape[:2]
        if min(h, w) < int(self.cfg.enroll_min_face_px):
            return

        emb = self.embed(face_bgr)
        if emb is None:
            return

        self._samples.append(emb)
        if len(self._samples) >= int(self.cfg.enroll_samples):
            m = np.mean(np.stack(self._samples, axis=0), axis=0)
            self._template = _l2norm(m.astype(np.float32))
            self.enrolled = True
            self.enrolling = False
            self._samples = []
            self.last_score = -1.0
            print("[FaceID] Enrollment complete -> enrolled=Y")

    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self.enabled or self._sess is None or self._in_name is None:
            return None

        cfg = self.cfg
        s = int(cfg.input_size)

        img = cv2.resize(face_bgr, (s, s), interpolation=cv2.INTER_LINEAR)
        if cfg.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = img.astype(np.float32)
        x = (x - cfg.mean) / cfg.std

        if self._layout == "NHWC":
            inp = x[None, :, :, :]  # (1,112,112,3)
        else:
            inp = np.transpose(x, (2, 0, 1))[None, :, :, :]  # (1,3,112,112)

        inp = np.ascontiguousarray(inp, dtype=np.float32)
        out = self._sess.run(None, {self._in_name: inp})[0]
        feat = np.asarray(out).reshape(-1).astype(np.float32)
        return _l2norm(feat)

    def match_score(self, face_bgr: np.ndarray) -> float:
        if not self.enrolled or self._template is None:
            self.last_score = -1.0
            return self.last_score
        emb = self.embed(face_bgr)
        if emb is None:
            self.last_score = -1.0
            return self.last_score
        self.last_score = float(np.dot(emb, self._template))
        return self.last_score

    def is_authorized(self, face_bgr: np.ndarray) -> bool:
        if not self.enrolled:
            self.last_score = -1.0
            return False
        s = self.match_score(face_bgr)
        return bool(s >= float(self.cfg.cosine_thr))
