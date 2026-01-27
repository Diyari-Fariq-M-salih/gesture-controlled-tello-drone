import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="dataset.csv")
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}

    print("Press numeric keys (0-9) to record that label. Press 'q' to quit.")
    for k, v in labels.items():
        print(f"  {k}: {v}")

    cap = cv2.VideoCapture(args.cam)
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["label"] + [f"p{i}_{c}" for i in range(21) for c in ("x", "y", "z")]
    file_exists = out_path.exists()

    with open(out_path, "a", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        if not file_exists:
            w.writerow(header)

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            pts = None
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).reshape(-1)
                h, wimg = frame.shape[:2]
                for p in lm:
                    cv2.circle(frame, (int(p.x * wimg), int(p.y * h)), 3, (0, 255, 0), -1)
                cv2.putText(frame, "HAND OK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            else:
                cv2.putText(frame, "NO HAND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            cv2.imshow("collect_dataset", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if ord("0") <= key <= ord("9"):
                lbl = int(chr(key))
                if lbl not in labels:
                    print(f"Label {lbl} not in labels.json")
                    continue
                if pts is None:
                    print("No hand detected; not recording.")
                    continue
                w.writerow([lbl, *pts.tolist()])
                print(f"Recorded: {lbl} ({labels[lbl]})")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
