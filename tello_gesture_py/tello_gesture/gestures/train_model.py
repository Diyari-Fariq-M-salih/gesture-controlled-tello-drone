import argparse
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="model.joblib")
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

    print("Labels:", labels)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(clf, args.out)
    print("Saved model:", args.out)

if __name__ == "__main__":
    main()
