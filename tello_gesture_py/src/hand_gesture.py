from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2
import mediapipe as mp

@dataclass
class HandDetection:
    has_hand: bool
    landmarks: Optional[np.ndarray]  # (21,3), normalized

class HandGesture:
    """MediaPipe Hands wrapper returning normalized landmarks."""

    def __init__(self, max_num_hands: int = 1):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def detect(self, bgr: np.ndarray) -> HandDetection:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            return HandDetection(False, None)
        lm = res.multi_hand_landmarks[0].landmark
        arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        return HandDetection(True, arr)
