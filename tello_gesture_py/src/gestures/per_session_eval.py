"""Within-session evaluation: each session trained and tested on itself.

Mirrors the protocol that produced the headline 99.7% figure, applied to all
three sessions independently, so the inflated baseline can be shown per session
rather than argued about in prose.

    python -m tello_gesture_py.src.gestures.per_session_eval
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .loso_eval import discover_sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default="outputs/metrics/per_session_eval.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = {}
    print(f"{'session':<9}{'train n':>9}{'test n':>8}{'accuracy':>10}{'macro F1':>10}")
    for key, path in discover_sessions().items():
        df = pd.read_csv(path)
        y = df["label"].astype(int).values
        X = df.drop(columns=["label"]).values.astype(np.float32)

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=args.seed, stratify=y)
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", probability=True, C=10.0, gamma="scale")),
        ])
        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)

        acc = float(accuracy_score(yte, yp))
        mf1 = float(f1_score(yte, yp, average="macro"))
        cm = confusion_matrix(yte, yp, labels=list(range(7)))
        out[key] = {
            "train_n": int(len(ytr)), "test_n": int(len(yte)),
            "accuracy": acc, "macro_f1": mf1,
            "confusion_matrix": cm.tolist(),
            "per_class_f1": f1_score(yte, yp, average=None, labels=list(range(7))).tolist(),
        }
        print(f"{key:<9}{len(ytr):>9}{len(yte):>8}{acc:>10.3f}{mf1:>10.3f}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\nwrote", args.out_json)


if __name__ == "__main__":
    main()
