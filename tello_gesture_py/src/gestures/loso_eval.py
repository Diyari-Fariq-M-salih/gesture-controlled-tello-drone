"""Leave-one-session-out evaluation of the gesture classifier.

Each fold trains on every session but one and tests on the held-out session, so
no train/test pairing has to be chosen by hand. Also emits the full
session-to-session transfer matrix.

With three sessions that matrix cannot separate illumination from background: A
is daylight and uniform, B and C are artificial and non-uniform, so the two
factors move together and "capture domain" is a bundle. A fourth session that is
daylight and non-uniform breaks the confound, and the pair summary below then
answers the question numerically -- mean transfer across pairs that share
illumination against pairs that share background -- instead of by reading the
matrix.

The three-session result is reported alongside the full one whenever more than
three sessions are present, so the figure the manuscript already carries stays
checkable against the extended one.

    python -m tello_gesture_py.src.gestures.loso_eval
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Every session is discovered from its filename, which is what lets a session
# be added without editing the evaluation that reports it. (Session A was
# renamed from dataset_features.csv to match on 2026-09-23.)


SESSION_REGISTRY = "data/sessions.json"


def discover_sessions():
    out = {}
    for p in sorted(glob.glob("data/processed/session?_features.csv")):
        key = os.path.basename(p)[len("session")]
        if os.path.exists(p):
            out[key] = p
    return dict(sorted(out.items()))


def session_meta():
    """Capture conditions per session, or {} if the registry is absent."""
    if not os.path.exists(SESSION_REGISTRY):
        return {}
    try:
        return json.load(open(SESSION_REGISTRY, encoding="utf-8")).get("sessions", {})
    except Exception:
        return {}


def pair_summary(transfer, meta):
    """Mean transfer grouped by what each ordered pair holds constant.

    Every ordered pair (train -> test) falls into exactly one cell of
    illumination-shared x background-shared. With A, B and C alone the
    off-diagonal cells are empty, because no pair shares one factor without the
    other; that emptiness is the confound, and it is reported rather than
    papered over.
    """
    # Groups are named for what CHANGES between the two sessions, because that
    # is what the group measures. A pair that shares illumination and differs in
    # background is the measurement of the background effect.
    groups = {"nothing_varies": [], "background_varies": [],
              "illumination_varies": [], "both_vary": []}
    detail = {}
    for tr, row in transfer.items():
        for te, acc in row.items():
            a, b = meta.get(tr), meta.get(te)
            if not a or not b:
                continue
            vi = a.get("illumination") != b.get("illumination")
            vb = a.get("background") != b.get("background")
            key = ("both_vary" if vi and vb else
                   "illumination_varies" if vi else
                   "background_varies" if vb else "nothing_varies")
            groups[key].append(acc)
            detail[f"{tr}->{te}"] = {"accuracy": round(acc, 4), "group": key,
                                     "illumination_varies": vi, "background_varies": vb}
    out = {"groups": {}, "pairs": detail}
    for k, v in groups.items():
        out["groups"][k] = {"n_pairs": len(v),
                            "mean_accuracy": round(float(np.mean(v)), 4) if v else None,
                            "min": round(float(np.min(v)), 4) if v else None,
                            "max": round(float(np.max(v)), 4) if v else None}
    bg, il = out["groups"]["background_varies"], out["groups"]["illumination_varies"]
    if bg["mean_accuracy"] is not None and il["mean_accuracy"] is not None:
        # Lower transfer means the change did more damage, so the dominant
        # factor is the one whose variation produces the lower mean.
        out["verdict"] = ("background dominates"
                          if bg["mean_accuracy"] < il["mean_accuracy"]
                          else "illumination dominates")
        out["gap"] = round(abs(il["mean_accuracy"] - bg["mean_accuracy"]), 4)
        out["background_varies_mean"] = bg["mean_accuracy"]
        out["illumination_varies_mean"] = il["mean_accuracy"]
    else:
        out["verdict"] = ("not identifiable: no session pair varies one factor "
                          "without the other")
        out["gap"] = None
    return out


def loso(data, keys):
    """Leave-one-session-out over the given session keys."""
    res = {}
    for held in keys:
        Xte, yte = data[held]
        others = [k for k in keys if k != held]
        Xtr = np.vstack([data[k][0] for k in others])
        ytr = np.concatenate([data[k][1] for k in others])
        yp = fit_predict(Xtr, ytr, Xte)
        res[held] = {
            "train_sessions": others, "train_n": int(len(ytr)), "test_n": int(len(yte)),
            "accuracy": float(accuracy_score(yte, yp)),
            "macro_f1": float(f1_score(yte, yp, average="macro")),
            "confusion_matrix": confusion_matrix(yte, yp, labels=list(range(7))).tolist(),
            "per_class_f1": f1_score(yte, yp, average=None, labels=list(range(7))).tolist(),
        }
    return res


def load(path):
    df = pd.read_csv(path)
    y = df["label"].astype(int).values
    X = df.drop(columns=["label"]).values.astype(np.float32)
    return X, y


def fit_predict(Xtr, ytr, Xte):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True, C=10.0, gamma="scale")),
    ])
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default="outputs/metrics/loso_eval.json")
    args = ap.parse_args()

    sessions = discover_sessions()
    data = {k: load(v) for k, v in sessions.items()}
    if len(data) < 3:
        raise SystemExit(f"need at least three sessions, found {sorted(data)}")
    keys = list(data)
    print("sessions: " + ", ".join(f"{k} ({sessions[k]})" for k in keys))
    print()

    meta = session_meta()
    out = {"loso": {}, "transfer": {}, "sessions": sessions, "session_meta": meta}

    print("=== leave-one-session-out ===")
    print(f"{'held out':<10}{'train n':>9}{'test n':>8}{'accuracy':>10}{'macro F1':>10}")
    out["loso"] = loso(data, keys)
    for held in keys:
        r = out["loso"][held]
        print(f"{held:<10}{r['train_n']:>9}{r['test_n']:>8}"
              f"{r['accuracy']:>10.3f}{r['macro_f1']:>10.3f}")
    out["loso_mean_accuracy"] = round(
        float(np.mean([out["loso"][k]["accuracy"] for k in keys])), 4)
    print(f"{'mean':<10}{'':>9}{'':>8}{out['loso_mean_accuracy']:>10.3f}")

    # The three-session result the manuscript already reports, kept alongside
    # the extended one so the two can be compared rather than swapped.
    if len(keys) > 3:
        base = [k for k in ("A", "B", "C") if k in data]
        if len(base) == 3:
            out["loso_ABC"] = loso(data, base)
            m = float(np.mean([out["loso_ABC"][k]["accuracy"] for k in base]))
            out["loso_ABC_mean_accuracy"] = round(m, 4)
            print()
            print("=== leave-one-session-out, sessions A-C only (as published) ===")
            for k in base:
                print(f"{k:<10}{'':>9}{'':>8}{out['loso_ABC'][k]['accuracy']:>10.3f}")
            print(f"{'mean':<10}{'':>9}{'':>8}{m:>10.3f}")

    print()
    print("=== full transfer matrix (train row -> test column, accuracy) ===")
    print(f"{'train':<8}" + "".join(f"{k:>10}" for k in keys))
    for tr in keys:
        row = {}
        cells = []
        for te in keys:
            if tr == te:
                cells.append("     --   ")
                continue
            yp = fit_predict(data[tr][0], data[tr][1], data[te][0])
            a = float(accuracy_score(data[te][1], yp))
            row[te] = a
            cells.append(f"{a:>10.3f}")
        out["transfer"][tr] = row
        print(f"{tr:<8}" + "".join(cells))

    ps = pair_summary(out["transfer"], meta)
    out["pair_summary"] = ps
    print()
    print("=== transfer by what varies between the pair ===")
    if not meta:
        print("  no data/sessions.json: cannot group pairs by condition")
    else:
        print(f"{'group':<20}{'pairs':>7}{'mean':>9}{'min':>8}{'max':>8}")
        for k, v in ps["groups"].items():
            if not v["n_pairs"]:
                print(f"{k:<20}{0:>7}{'--':>9}{'--':>8}{'--':>8}")
                continue
            print(f"{k:<20}{v['n_pairs']:>7}{v['mean_accuracy']:>9.3f}"
                  f"{v['min']:>8.3f}{v['max']:>8.3f}")
        if ps["gap"] is not None:
            print(f"\n  varying background costs more than varying illumination"
                  if ps["verdict"] == "background dominates" else
                  f"\n  varying illumination costs more than varying background")
            print(f"  verdict: {ps['verdict']}  (gap {ps['gap']:.3f})")
        else:
            print(f"\n  verdict: {ps['verdict']}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print()
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
