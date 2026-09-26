"""Validate the frozen SVM artefact before it is flown.

The artefact was previously checked only against the within-session figure
(0.9971 vs 0.997). That is the leaky protocol this paper exists to criticise and
it cannot establish that the retrained file is the characterised model: a random
split over near-duplicate frames will score ~1.0 for almost any competent model.

Two of the four requested checks cannot be performed as written, for reasons that
are properties of the repository rather than of the artefact:

  * `per_session_eval.py` and `loso_eval.py` never load a joblib. They retrain
    from the session feature CSVs on every invocation, so the leave-one-session-
    out and transfer figures are not outputs of any saved artefact. Running them
    tests whether this environment still reproduces the published numbers, which
    is worth knowing, but it is not a test of the frozen file. Reported as
    ENVIRONMENT rather than as artefact validation.

  * the 2947 ground-pipeline frames behind Table VI retain only derived
    quantities and the model's own outputs -- `bbox_area`, `delta_s`,
    `svm_label`, `svm_prob` -- not the 63-dimensional landmark vectors. The
    frozen model therefore cannot be re-run on them offline. What can be
    verified is that the recorded predictions reproduce 0.783, 0.809 and the two
    off-diagonal cells, which validates the analysis chain rather than the
    artefact, and that the artefact used was this one.

The check that does bear directly on the artefact is per-prediction agreement
against the original `model-400.joblib` on every held-out set, and that is run
here at full strength.

    python -m tello_gesture_py.scripts.validate_frozen_model
"""

import argparse
import hashlib
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tello_gesture_py.src.gesture_classifier import FROZEN_SVM_SHA256  # noqa: E402

CLASSES = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN", "FORWARD", "BACK"]
FEATURES = {
    "A": "data/processed/sessionA_features.csv",
    "B": "data/processed/sessionB_features.csv",
    "C": "data/processed/sessionC_features.csv",
    "D": "data/processed/sessionD_features.csv",
}
PUBLISHED_LOSO = {"A": 0.793, "B": 0.882, "C": 0.978, "D": 0.988}
PUBLISHED_LOSO_MEAN = 0.910
PUBLISHED_WITHIN = {"A": 0.997, "B": 0.988, "C": 1.000, "D": 1.000}
PUBLISHED_TABLE_VI = {"per_frame": 0.783, "per_hold": 0.809,
                      "RIGHT_to_BACK": 0.72, "FORWARD_to_BACK": 0.71}
TOL_ACC = 0.002
TOL_DISAGREE = 0.01

RESULTS = []


def record(check, kind, ok, detail):
    RESULTS.append({"check": check, "kind": kind, "ok": ok, "detail": detail})
    mark = "PASS" if ok else ("----" if kind == "INFO" else "FAIL")
    print(f"  [{mark}] {check}: {detail}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_xy(path):
    d = pd.read_csv(path)
    y = d["label"].astype(int).values
    X = d.drop(columns=["label"]).values.astype(np.float32)
    return X, y


# ---------------------------------------------------------------- checks
def check_environment():
    """Does this environment still reproduce the published Section V-C numbers?

    Not a test of the artefact -- these scripts retrain -- but if the answer is
    no, every figure in Section V-C is in question regardless of which file is
    flown.
    """
    print("\nENVIRONMENT -- do the published evaluations still reproduce here?")
    from tello_gesture_py.src.gestures import loso_eval, per_session_eval  # noqa: F401
    import subprocess
    for mod, out in (("per_session_eval", "outputs/metrics/per_session_eval.json"),
                     ("loso_eval", "outputs/metrics/loso_eval.json")):
        subprocess.run([sys.executable, "-m", f"tello_gesture_py.src.gestures.{mod}"],
                       capture_output=True)

    w = json.load(open("outputs/metrics/per_session_eval.json", encoding="utf-8"))
    bad = {k: round(w[k]["accuracy"], 3) for k in PUBLISHED_WITHIN
           if abs(w[k]["accuracy"] - PUBLISHED_WITHIN[k]) > TOL_ACC}
    record("within-session reproduces", "ENVIRONMENT", not bad,
           "all four match" if not bad else f"drift: {bad}")

    j = json.load(open("outputs/metrics/loso_eval.json", encoding="utf-8"))
    L = j["loso"]
    bad = {k: round(L[k]["accuracy"], 3) for k in PUBLISHED_LOSO
           if abs(L[k]["accuracy"] - PUBLISHED_LOSO[k]) > TOL_ACC}
    mean = float(np.mean([L[k]["accuracy"] for k in L]))
    record("leave-one-session-out reproduces", "ENVIRONMENT",
           not bad and abs(mean - PUBLISHED_LOSO_MEAN) <= TOL_ACC,
           f"mean {mean:.3f} vs {PUBLISHED_LOSO_MEAN}"
           + ("" if not bad else f", drift: {bad}"))

    T = j["transfer"]
    pairs = {f"{a}->{b}": v for a, row in T.items() for b, v in row.items()}
    record("transfer matrix present", "ENVIRONMENT", len(pairs) == 12,
           f"{len(pairs)} ordered pairs computed")


def check_artefact_agreement(frozen, original):
    """Per-prediction agreement between the two candidate artefacts.

    The decisive check. If the retrain changed the model, it shows up here as
    disagreement on individual samples even when aggregate accuracies match.
    """
    print("\nARTEFACT -- frozen vs original, per prediction")
    import joblib
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fm = joblib.load(frozen)
        om = joblib.load(original)

    total = diff = 0
    per_session = {}
    for k, path in FEATURES.items():
        if not os.path.exists(path):
            continue
        X, y = load_xy(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf, po = fm.predict(X), om.predict(X)
        d = int((pf != po).sum())
        per_session[k] = {"n": len(y), "disagree": d, "frac": round(d / len(y), 5),
                          "acc_frozen": round(float((pf == y).mean()), 4),
                          "acc_original": round(float((po == y).mean()), 4)}
        total += len(y)
        diff += d
        print(f"    session {k}: n={len(y):5d}  disagree={d:5d} "
              f"({d/len(y):6.2%})   acc {per_session[k]['acc_frozen']:.4f} "
              f"vs {per_session[k]['acc_original']:.4f}")

    frac = diff / total if total else 1.0
    record("per-prediction agreement", "ARTEFACT", frac < TOL_DISAGREE,
           f"{diff}/{total} samples differ ({frac:.2%}), threshold {TOL_DISAGREE:.0%}")
    return per_session, frac


def check_determinism(frozen):
    """Retraining with the same seed and pipeline must give the same predictions."""
    print("\nARTEFACT -- is the frozen file reproducible from its own recipe?")
    import subprocess
    import tempfile
    import joblib
    tmp = os.path.join(tempfile.gettempdir(), "_refit.joblib")
    r = subprocess.run(
        [sys.executable, "-m", "tello_gesture_py.src.gestures.train_model",
         "--dataset", FEATURES["A"], "--labels", "data/labels/labels_example.json",
         "--out", tmp, "--metrics_out", os.path.join(tempfile.gettempdir(), "_m.json"),
         "--cm_png_out", os.path.join(tempfile.gettempdir(), "_cm.png")],
        capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp):
        record("retrain is deterministic", "ARTEFACT", False,
               f"retrain failed: {r.stderr[-160:]}")
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b = joblib.load(frozen), joblib.load(tmp)
    X, _ = load_xy(FEATURES["A"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = int((a.predict(X) != b.predict(X)).sum())
    record("retrain is deterministic", "ARTEFACT", d == 0,
           f"{d}/{len(X)} predictions differ from a fresh fit of the same recipe")
    os.remove(tmp)


def check_table_vi():
    """Do the recorded predictions still yield Table VI's figures?

    Validates the analysis chain. The artefact itself cannot be re-run on these
    frames because the landmarks were not logged.
    """
    print("\nANALYSIS -- Table VI from the recorded predictions")
    p = "outputs/runs/20260920-113736_capture_gesture-svm/op_gesture.csv"
    if not os.path.exists(p):
        record("Table VI reproduces", "ANALYSIS", False, "capture log missing")
        return
    d = pd.read_csv(p)
    d = d[d["phase"] == "hold"]
    ok_frame = (d["svm_name"] == d["cued_name"])
    per_frame = float(ok_frame.mean())

    holds, hit = 0, 0
    for _, g in d.groupby(["round", "cued_label"]):
        holds += 1
        v = g["svm_name"].dropna().mode()
        if len(v) and v.iloc[0] == g["cued_name"].iloc[0]:
            hit += 1
    per_hold = hit / holds if holds else 0.0

    def cell(cue, got):
        g = d[d["cued_name"] == cue]
        return float((g["svm_name"] == got).mean()) if len(g) else float("nan")

    r2b, f2b = cell("RIGHT", "BACK"), cell("FORWARD", "BACK")
    ok = (abs(per_frame - PUBLISHED_TABLE_VI["per_frame"]) <= TOL_ACC
          and abs(per_hold - PUBLISHED_TABLE_VI["per_hold"]) <= TOL_ACC
          and abs(r2b - PUBLISHED_TABLE_VI["RIGHT_to_BACK"]) <= 0.01
          and abs(f2b - PUBLISHED_TABLE_VI["FORWARD_to_BACK"]) <= 0.01)
    record("Table VI reproduces", "ANALYSIS", ok,
           f"per-frame {per_frame:.3f}, per-hold {per_hold:.3f} over {holds} holds, "
           f"RIGHT->BACK {r2b:.2f}, FORWARD->BACK {f2b:.2f}")

    record("landmarks retained for offline re-scoring", "INFO", False,
           "op_gesture.csv holds derived quantities only, so the frozen model "
           "cannot be re-run on these frames; provenance rests on file times "
           "and resolution order")


def check_provenance(frozen):
    """Which artefact produced the 0.783 capture?"""
    print("\nPROVENANCE -- which file produced Table VI")
    m = json.load(open(
        "outputs/runs/20260920-113736_capture_gesture-svm/manifest.json", encoding="utf-8"))
    recorded = (m.get("model") or {}).get("gesture_model")
    cap = m.get("started_at", 0)
    mt = os.path.getmtime(frozen)
    record("frozen artefact predates the capture", "PROVENANCE", mt < cap,
           f"model mtime {mt:.0f} < capture {cap:.0f}"
           if mt < cap else "model is NEWER than the capture")
    record("capture recorded a bare model name", "INFO", False,
           f"manifest says {recorded!r}; no hash was recorded at capture time "
           "(the harness records one now)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default="models/production/model.joblib")
    ap.add_argument("--original", default="models/experiments/model-400.joblib")
    ap.add_argument("--out_json", default="outputs/metrics/frozen_model_validation.json")
    args = ap.parse_args()

    print("=" * 72)
    print("  Frozen model validation")
    print("=" * 72)
    fh, oh = sha256(args.frozen), sha256(args.original)
    print(f"  frozen   {args.frozen}\n           {fh}")
    print(f"  original {args.original}\n           {oh}")
    record("frozen hash matches the pinned constant", "ARTEFACT",
           fh == FROZEN_SVM_SHA256,
           "matches gesture_classifier.FROZEN_SVM_SHA256" if fh == FROZEN_SVM_SHA256
           else "DOES NOT MATCH the pinned constant")

    check_environment()
    per_session, frac = check_artefact_agreement(args.frozen, args.original)
    check_determinism(args.frozen)
    check_table_vi()
    check_provenance(args.frozen)

    hard = [r for r in RESULTS if r["kind"] != "INFO"]
    failed = [r for r in hard if not r["ok"]]
    print("\n" + "=" * 72)
    print(f"  {len(hard) - len(failed)}/{len(hard)} checks passed")
    for r in failed:
        print(f"  FAILED  [{r['kind']}] {r['check']}: {r['detail']}")
    if failed:
        print("\n  Do not fly. If agreement failed, pin scikit-learn 1.7.2 in a")
        print("  separate environment and freeze the original artefact instead.")
    print("=" * 72)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"frozen": args.frozen, "frozen_sha256": fh,
                   "original": args.original, "original_sha256": oh,
                   "disagreement_fraction": frac,
                   "per_session": per_session, "results": RESULTS}, f, indent=2)
    print(f"wrote {args.out_json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
