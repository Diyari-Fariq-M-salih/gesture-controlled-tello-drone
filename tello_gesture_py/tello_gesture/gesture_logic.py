from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class GestureResult:
    name: str
    confidence: float

"""Simple rule-based hand gesture classifier. without training for fast prototyping baby!!!!!!"""
class RuleBasedGesture:
    """
    - LEFT/RIGHT/UP/DOWN from index direction (mcp=5, tip=8)
    - FORWARD/BACK from relative hand scale change (bbox area).
    """

    def __init__(self, dir_thr: float = 0.10, scale_thr: float = 0.18, ema_alpha: float = 0.35):
        self.dir_thr = dir_thr
        self.scale_thr = scale_thr
        self.ema_alpha = ema_alpha
        self._ema: Optional[np.ndarray] = None
        self._last_scale: Optional[float] = None

    def _smooth(self, lm: np.ndarray) -> np.ndarray:
        if self._ema is None:
            self._ema = lm.copy()
        else:
            self._ema = self.ema_alpha * lm + (1.0 - self.ema_alpha) * self._ema
        return self._ema

    def _scale(self, lm: np.ndarray) -> float:
        xs = lm[:, 0]
        ys = lm[:, 1]
        w = float(xs.max() - xs.min())
        h = float(ys.max() - ys.min())
        return max(w * h, 1e-6)

    def predict(self, lm: np.ndarray) -> GestureResult:
        lm = self._smooth(lm)

        mcp = lm[5]
        tip = lm[8]
        dx = float(tip[0] - mcp[0])
        dy = float(tip[1] - mcp[1])

        parts = []
        if dy < -self.dir_thr:
            parts.append("UP")
        elif dy > self.dir_thr:
            parts.append("DOWN")
        if dx < -self.dir_thr:
            parts.append("LEFT")
        elif dx > self.dir_thr:
            parts.append("RIGHT")

        sc = self._scale(lm)
        if self._last_scale is not None:
            rel = (sc - self._last_scale) / max(self._last_scale, 1e-6)
            if rel > self.scale_thr:
                parts.append("FORWARD")
            elif rel < -self.scale_thr:
                parts.append("BACK")
        self._last_scale = sc

        if not parts:
            return GestureResult("CENTER", 0.5)

        conf = min(1.0, 0.5 + 2.0 * max(abs(dx), abs(dy)))
        return GestureResult("-".join(parts), conf)
