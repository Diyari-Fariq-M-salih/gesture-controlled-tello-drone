from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2
import mediapipe as mp

@dataclass
class HandDetection:
    has_hand: bool
    landmarks: Optional[np.ndarray]  # (21,3), normalized; the first hand detected
    # Every detected hand, index-aligned with its MediaPipe handedness. With the
    # deployed max_num_hands=1 this is just [landmarks]; the hand-face associator
    # (hand_association.py) is what uses more than one.
    hands: list = field(default_factory=list)
    # MediaPipe Hands labels handedness assuming a MIRRORED (selfie) image, so on
    # the raw Tello feed a real right hand is labelled 'Left'. See
    # HandAssociator._resolve_sides().
    handedness: list = field(default_factory=list)        # 'Left' / 'Right' per hand
    handedness_score: list = field(default_factory=list)  # [0,1] per hand

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
        hands, labels, scores = [], [], []
        handedness = res.multi_handedness or []
        for i, hand in enumerate(res.multi_hand_landmarks):
            hands.append(np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32))
            if i < len(handedness) and handedness[i].classification:
                cls = handedness[i].classification[0]
                labels.append(cls.label)
                scores.append(float(cls.score))
            else:
                labels.append(None)
                scores.append(None)
        # `landmarks` stays the first hand, exactly as the deployed single-hand
        # path has always read it.
        return HandDetection(True, hands[0], hands, labels, scores)
