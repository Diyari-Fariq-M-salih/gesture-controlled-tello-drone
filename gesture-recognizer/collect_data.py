import os
import time
import cv2
import numpy as np
import mediapipe as mp
from utils import ensure_dirs, save_gesture_labels, build_feature_vector

# Updated gesture set (8 classes)
GESTURES = [
    "POINT_FORWARD",
    "PEACE_BACK",
    "THUMBS_LEFT",
    "PINKY_RIGHT",
    "OPEN_UP",
    "FIST_DOWN",
    "OKAY_LAND",
    "LOVE_ROTATE",
]

def main():
    ensure_dirs()
    save_gesture_labels(GESTURES)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    current_label_idx = 0
    samples = []  # list of (features64, label_idx)
    last_msg = ""

    print("=== Data Collection ===")
    print("Keys:")
    print("  1..8 : choose gesture class")
    print("  S    : save sample")
    print("  Q    : quit and write file")
    print("\nGesture classes:")
    for i, g in enumerate(GESTURES, start=1):
        print(f"  {i}: {g}")

    out_path = os.path.join("data", "raw", f"samples_{int(time.time())}.npz")

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

            frame = cv2.flip(frame, 1)  # selfie view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            h, w = frame.shape[:2]
            feat = None

            if res.multi_hand_landmarks and res.multi_handedness:
                hand_lms = res.multi_hand_landmarks[0]
                handed = res.multi_handedness[0].classification[0].label  # "Left"/"Right"

                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                lm_xyz = np.array([(p.x, p.y, p.z) for p in hand_lms.landmark], dtype=np.float32)
                feat = build_feature_vector(lm_xyz, handed)  # (64,)

                # Bounding box
                xs = [int(p.x * w) for p in hand_lms.landmark]
                ys = [int(p.y * h) for p in hand_lms.landmark]
                x1 = max(min(xs) - 20, 0)
                y1 = max(min(ys) - 20, 0)
                x2 = min(max(xs) + 20, w)
                y2 = min(max(ys) + 20, h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame, f"Hand: {handed}", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # UI overlay
            label_name = GESTURES[current_label_idx]
            cv2.putText(frame, f"Class [{current_label_idx+1}/8]: {label_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"Samples: {len(samples)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if last_msg:
                cv2.putText(frame, last_msg, (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

            cv2.imshow("Collect Data (S=save, 1..8=class, Q=quit)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), ord('Q')]:
                break

            # Choose class 1..8
            if ord('1') <= key <= ord('8'):
                idx = key - ord('1')
                current_label_idx = idx
                last_msg = f"Selected {current_label_idx+1}: {GESTURES[current_label_idx]}"

            # Save sample
            if key in [ord('s'), ord('S')]:
                if feat is None:
                    last_msg = "No hand detected — can't save."
                else:
                    samples.append((feat, current_label_idx))
                    last_msg = f"Saved sample for {GESTURES[current_label_idx]}"

    cap.release()
    cv2.destroyAllWindows()

    if len(samples) == 0:
        print("No samples collected. Nothing saved.")
        return

    X = np.stack([s[0] for s in samples], axis=0).astype(np.float32)  # (N,64)
    y = np.array([s[1] for s in samples], dtype=np.int64)             # (N,)

    np.savez_compressed(out_path, X=X, y=y, gestures=np.array(GESTURES, dtype=object))
    print(f"Saved {len(samples)} samples to: {out_path}")
    print("Data tip:")
    print("- Collect ~300–800 samples per class")
    print("- Vary distance, rotation, lighting")
    print("- For PEACE_BACK: keep the BACK of the hand facing camera while doing peace sign")

if __name__ == "__main__":
    main()
