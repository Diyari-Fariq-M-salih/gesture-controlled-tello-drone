import argparse
import os
import pandas as pd
import numpy as np
import cv2

def extract_hand_landmarks_mediapipe(image_bgr, hands):
    import mediapipe as mp

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(image_rgb)

    if not res.multi_hand_landmarks:
        return None

    # Take the first detected hand
    lm = res.multi_hand_landmarks[0].landmark

    # 21 landmarks, each has x,y,z (normalized)
    feats = []
    for p in lm:
        feats.extend([p.x, p.y, p.z])

    return np.array(feats, dtype=np.float32)  # length 63


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="CSV with columns: path,label")
    ap.add_argument("--out", default="dataset_features.csv", help="Output numeric-features CSV")
    ap.add_argument("--max_rows", type=int, default=0, help="0 = all rows (debug otherwise)")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset)

    # Accept either 'path' or first column being paths
    if "path" not in df.columns:
        # If your csv is [path,label] but unnamed, rename safely
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: "path", df.columns[1]: "label"})
        else:
            raise ValueError("Dataset must contain 'path' and 'label' columns.")

    paths = df["path"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    if args.max_rows and args.max_rows > 0:
        paths = paths[: args.max_rows]
        labels = labels[: args.max_rows]

    import mediapipe as mp
    mp_hands = mp.solutions.hands

    X = []
    y = []
    dropped = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands:
        for p, lab in zip(paths, labels):
            if not os.path.exists(p):
                dropped += 1
                continue

            img = cv2.imread(p)
            if img is None:
                dropped += 1
                continue

            feats = extract_hand_landmarks_mediapipe(img, hands)
            if feats is None:
                dropped += 1
                continue

            X.append(feats)
            y.append(lab)

    if not X:
        raise RuntimeError("No usable samples: MediaPipe did not detect hands in any images.")

    X = np.stack(X, axis=0)
    out_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    out_df["label"] = y
    out_df.to_csv(args.out, index=False)

    total = len(paths)
    kept = len(y)
    print(f"Done. Total={total}, kept={kept}, dropped(no hand/missing/bad)={dropped}")
    print(f"Wrote: {args.out} (shape {out_df.shape})")


if __name__ == "__main__":
    main()
