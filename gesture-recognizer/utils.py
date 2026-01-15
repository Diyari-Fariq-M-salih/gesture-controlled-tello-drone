import os
import json
import numpy as np

GESTURE_FILE = os.path.join("data", "gesture_labels.json")

def ensure_dirs():
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    os.makedirs("models", exist_ok=True)

def save_gesture_labels(labels):
    ensure_dirs()
    with open(GESTURE_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

def load_gesture_labels():
    if not os.path.exists(GESTURE_FILE):
        return None
    with open(GESTURE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_landmarks_xyz(lm_xyz: np.ndarray) -> np.ndarray:
    """
    lm_xyz: (21,3) MediaPipe normalized landmarks (x,y,z).
    Returns a flattened 63-dim vector with translation + scale normalization.
      - Translate so wrist (0) is at origin
      - Scale by max distance to wrist (in 3D)
    """
    lm = lm_xyz.astype(np.float32).copy()
    wrist = lm[0].copy()
    lm -= wrist  # translation invariance

    norms = np.linalg.norm(lm, axis=1)
    scale = float(np.max(norms)) if float(np.max(norms)) > 1e-6 else 1.0
    lm /= scale

    return lm.reshape(-1)  # 63

def handedness_bit(label: str) -> float:
    """
    MediaPipe handedness label is typically 'Left' or 'Right'.
    Return 0.0 for Left, 1.0 for Right (stable numeric feature).
    """
    return 1.0 if label.strip().lower() == "right" else 0.0

def build_feature_vector(lm_xyz: np.ndarray, hand_label: str) -> np.ndarray:
    """
    Returns 64-dim feature vector: 63 landmark features + 1 handedness bit.
    """
    feat63 = normalize_landmarks_xyz(lm_xyz)  # (63,)
    hb = np.array([handedness_bit(hand_label)], dtype=np.float32)  # (1,)
    return np.concatenate([feat63, hb], axis=0).astype(np.float32)  # (64,)
