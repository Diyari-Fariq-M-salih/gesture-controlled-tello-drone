import json
from dataclasses import dataclass
from typing import Dict
from pathlib import Path

import numpy as np
import joblib


# src/ is at: project-root/tello_gesture_py/src  -> parents[2] = tello_gesture_py, parents[3] = project-root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_model_path(p: str) -> str:
    P = Path(p)
    if P.is_absolute():
        return str(P)

    # if it's just a filename, prefer the new structure
    if P.parent == Path("."):
        for candidate in [
            PROJECT_ROOT / "models" / "production" / P.name,
            PROJECT_ROOT / "models" / "experiments" / P.name,
            PROJECT_ROOT / "models" / P.name,
        ]:
            if candidate.exists():
                return str(candidate)
        # fallback if not found yet
        return str(PROJECT_ROOT / "models" / "experiments" / P.name)

    # otherwise treat it as relative to project root
    return str(PROJECT_ROOT / P)


def _resolve_labels_path(p: str) -> str:
    P = Path(p)
    if P.is_absolute():
        return str(P)

    if P.parent == Path("."):
        candidate = PROJECT_ROOT / "data" / "labels" / P.name
        if candidate.exists():
            return str(candidate)
        return str(PROJECT_ROOT / "data" / "labels" / P.name)

    return str(PROJECT_ROOT / P)


@dataclass
class GestureResult:
    name: str
    confidence: float


class TrainedClassifier:
    """Loads a scikit-learn model trained on flattened (x,y,z)*21 landmarks."""

    def __init__(self, model_path: str, labels_path: str):
        model_path = _resolve_model_path(model_path)
        labels_path = _resolve_labels_path(labels_path)

        self.model = joblib.load(model_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.id_to_name: Dict[int, str] = {int(k): v for k, v in raw.items()}

    def predict(self, lm: np.ndarray) -> GestureResult:
        x = lm[:, :3].reshape(1, -1)  # (1, 63)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
            cls = int(np.argmax(proba))
            conf = float(np.max(proba))
        else:
            cls = int(self.model.predict(x)[0])
            conf = 0.6
        return GestureResult(self.id_to_name.get(cls, "UNKNOWN"), conf)