from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2
import mediapipe as mp

@dataclass
class HandDetection:
    has_hand: bool
    landmarks: Optional[list[np.ndarray]]  # list of (21,3) arrays, one per hand

class HandGesture:
    """MediaPipe Hands wrapper returning normalized landmarks."""

    def __init__(self, max_num_hands: int = 2):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.max_num_hands = max_num_hands
    
    def detect(self, bgr: np.ndarray) -> HandDetection:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self._hands.process(rgb)
        if not res.multi_hand_landmarks:
            return HandDetection(False, None)
        # list of arrays of 21 (x,y,z) normalized landmarks, one per hand
        lm_list = []
        num_hands = min(len(res.multi_hand_landmarks), self.max_num_hands)
        for i in range(num_hands):
            lm = res.multi_hand_landmarks[i].landmark
            arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            lm_list.append(arr)
        return HandDetection(True, lm_list)
    
    def detect_auth(self, bgr: np.ndarray, auth_bbox) -> HandDetection:
        """If multiple hands are detected, returns the one closest to the auth_bbox center (if within threshold).
        auth_bbox: (x,y,w,h) in normalized coords [0..1] relative to the input image size.
        """
        # Validate auth_bbox
        if not auth_bbox or len(auth_bbox) != 4:
            print("Invalid auth_bbox provided to detect_auth. Expected (x,y,w,h) in normalized coords.")
            return HandDetection(False, None)
        
        det = self.detect(bgr)
        if det.has_hand and len(det.landmarks) == 1:
            return det
        elif det.has_hand and len(det.landmarks) > 1:
            # find the hand closest to the auth_bbox center and return it if it's close enough
            auth_cx = auth_bbox[0] + auth_bbox[2] / 2
            auth_cy = auth_bbox[1] + auth_bbox[3] / 2
            best_idx = None
            best_dist = float('inf')
            for i, lm in enumerate(det.landmarks):
                # hand_cx = np.mean(lm[:,0])
                # hand_cy = np.mean(lm[:,1])
                hand_cx = lm[0,0]  # use wrist point as hand center
                hand_cy = lm[0,1]
                # use L2 distance in normalized coords
                dist = np.linalg.norm([(hand_cx - auth_cx), (hand_cy - auth_cy)])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            # threshold distance (in normalized coords) to consider it a match
            # if best_dist < 0.2:  # Increased from 0.1 to 0.2 for more lenient matching
            #     print(f"Selected hand {best_idx} for control based on proximity to authorized face (distance: {best_dist:.3f}).")
            #     return HandDetection(True, [det.landmarks[best_idx]])
            # print(f"Selected hand {best_idx} for control based on proximity to authorized face (distance: {best_dist:.3f}).")
            return HandDetection(True, [det.landmarks[best_idx]])
            # else:
            #     print(f"No hand close enough to authorized face. Best distance: {best_dist:.3f} (threshold: 0.2)")
        
        # Always return a HandDetection object (never None)
        return HandDetection(False, None)