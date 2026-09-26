"""Cross-session gesture classification evaluation.

Trains the SVM on one recording session's feature CSV and evaluates on a
*different* session's feature CSV (different day / lighting / re-enrollment),
to measure real generalization rather than in-session train/test-split
accuracy. Requested by the ARO content review (section 3). Superseded for the
paper by loso_eval.py, which runs every pairing; this produced the pairwise
outputs/metrics/cross_session_{AB,AC,BC}.json.

Usage (one line in cmd.exe):
    python -m tello_gesture_py.src.gestures.cross_session_eval
        --train_features data/processed/sessionA_features.csv
        --test_features  data/processed/sessionB_features.csv
        --labels data/labels/labels_example.json
        --out_json outputs/metrics/cross_session_AB.json

Both feature CSVs must have been produced by images_to_features.py (columns
f0..f62 + label) so the feature layout matches the in-session model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _load_xy(csv_path: str):
    df = pd.read_csv(csv_path)
    y = df["label"].astype(int).values
    X = df.drop(columns=["label"]).values.astype(np.float32)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_features", required=True, help="Session A dataset_features.csv (used for training)")
    ap.add_argument("--test_features", required=True, help="Session B dataset_features.csv (different session, used for testing)")
    ap.add_argument("--labels", required=True, help="labels.json mapping numeric id -> name")
    ap.add_argument("--out_json", default="outputs/metrics/cross_session_eval.json")
    args = ap.parse_args()

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}

    X_train, y_train = _load_xy(args.train_features)
    X_test, y_test = _load_xy(args.test_features)

    # Same pipeline/hyperparameters as the in-session model (train_model.py)
    # so results are directly comparable to the reported in-session numbers.
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, C=10.0, gamma="scale")),
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Train: {args.train_features} (N={len(y_train)})")
    print(f"Test:  {args.test_features} (N={len(y_test)}, different session)")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", cm)
    print(f"Cross-session accuracy: {acc:.4f}")

    out = {
        "train_features": args.train_features,
        "test_features": args.test_features,
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "accuracy": acc,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
