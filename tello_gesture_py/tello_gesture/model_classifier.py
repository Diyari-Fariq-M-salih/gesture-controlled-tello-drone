import json
from dataclasses import dataclass
from typing import Dict
import numpy as np
import joblib

@dataclass
class GestureResult:
    name: str
    confidence: float

class TrainedClassifier:
    """Loads a scikit-learn model trained on flattened (x,y,z)*21 landmarks."""

    def __init__(self, model_path: str, labels_path: str):
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
