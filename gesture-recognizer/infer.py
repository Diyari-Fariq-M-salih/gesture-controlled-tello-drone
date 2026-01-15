import time
import os
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from utils import build_feature_vector

IN_DIM = 64

# MUST MATCH train.py EXACTLY
class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-9)

def main():
    model_path = os.path.join("models", "model.pth")
    if not os.path.exists(model_path):
        raise RuntimeError("models/model.pth not found. Run: python train.py")
    if os.path.getsize(model_path) < 1024:
        raise RuntimeError("models/model.pth looks empty/corrupted. Delete it and rerun: python train.py")

    ckpt = torch.load(model_path, map_location="cpu")
    gestures = ckpt["gestures"]
    num_classes = ckpt["num_classes"]
    in_dim = ckpt.get("in_dim", IN_DIM)

    if in_dim != IN_DIM:
        raise RuntimeError(f"Model expects in_dim={in_dim}, but infer.py uses IN_DIM={IN_DIM}")

    model = MLP(in_dim=IN_DIM, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    last_pred = None
    last_print_time = 0.0
    PRINT_COOLDOWN = 0.25
    CONF_THRESH = 0.75  # adjust 0.60–0.85 depending on data quality

    print("=== Live Gesture Recognition ===")
    print("Press Q to quit.")
    print("Classes:", gestures)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            display_text = "No hand"
            conf_text = ""
            hand_text = ""

            if res.multi_hand_landmarks and res.multi_handedness:
                hand_lms = res.multi_hand_landmarks[0]
                handed = res.multi_handedness[0].classification[0].label  # "Left"/"Right"
                hand_text = f"Hand: {handed}"

                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                lm_xyz = np.array([(p.x, p.y, p.z) for p in hand_lms.landmark], dtype=np.float32)
                feat = build_feature_vector(lm_xyz, handed)  # (64,)

                with torch.no_grad():
                    logits = model(torch.from_numpy(feat).float().unsqueeze(0))  # (1,C)
                    logits_np = logits.numpy().squeeze(0)
                    probs = softmax_np(logits_np)
                    pred_idx = int(np.argmax(probs))
                    pred_conf = float(probs[pred_idx])

                conf_text = f"{pred_conf:.2f}"

                if pred_conf >= CONF_THRESH:
                    pred_label = gestures[pred_idx]
                    display_text = pred_label

                    now = time.time()
                    if (pred_label != last_pred) and (now - last_print_time > PRINT_COOLDOWN):
                        print(f"[{time.strftime('%H:%M:%S')}] {pred_label}  conf={pred_conf:.2f}  ({handed})")
                        last_pred = pred_label
                        last_print_time = now
                else:
                    display_text = "UNSURE"

            cv2.putText(frame, f"Pred: {display_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            if conf_text:
                cv2.putText(frame, f"Conf: {conf_text}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            if hand_text:
                cv2.putText(frame, hand_text, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow("Gesture Recognizer (Q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
