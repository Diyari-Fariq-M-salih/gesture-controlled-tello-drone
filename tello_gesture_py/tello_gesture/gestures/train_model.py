import argparse
import json
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt


def save_confusion_matrix_png(cm: np.ndarray, outpath: Path, class_names: list[str] | None = None):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    if class_names:
        ticks = np.arange(len(class_names))
        plt.xticks(ticks, class_names, rotation=45, ha="right")
        plt.yticks(ticks, class_names)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="model.joblib")

    # NEW:
    ap.add_argument("--metrics_out", default="training_metrics.json")
    ap.add_argument("--cm_png_out", default="confusion_matrix.png")

    args = ap.parse_args()

    with open(args.labels, "r", encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}

    df = pd.read_csv(args.dataset)
    y = df["label"].astype(int).values
    X = df.drop(columns=["label"]).values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, C=10.0, gamma="scale"))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    acc = float(accuracy_score(y_test, y_pred))

    print("Labels:", labels)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", cm)

    # Save model
    joblib.dump(clf, args.out)
    print("Saved model:", args.out)

    # Save confusion matrix PNG + metrics JSON
    class_names = [labels[i] for i in sorted(labels.keys())] if labels else None
    save_confusion_matrix_png(cm, Path(args.cm_png_out), class_names=class_names)

    metrics = {
        "accuracy": acc,
        "report": report_dict,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "dataset": args.dataset,
        "model_out": args.out,
    }
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved metrics:", args.metrics_out)
    print("Saved confusion matrix image:", args.cm_png_out)


if __name__ == "__main__":
    main()
