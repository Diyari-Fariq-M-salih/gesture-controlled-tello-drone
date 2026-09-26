"""Analysis of the cued ground capture: operational gesture accuracy.

Consumes `op_gesture.csv` written by scripts/operational_gesture_eval.py and
reports, for each classifier that the capture recorded:

  * operational accuracy over every hold-window frame, and accuracy over the
    frames where a hand was found, reported separately. They answer different
    questions -- an undetected hand issues no command, a misread hand issues the
    wrong one -- and a single number hides which dominates.
  * frame-detection rate, the fraction of cued frames in which MediaPipe found a
    hand at all.
  * a row-normalised confusion matrix in the style of Figure 1, with an extra
    column for no-detection, so it can sit beside the within-session and
    held-out matrices.
  * per-class F1.
  * the spurious FORWARD/BACK rate on frames cued LEFT/RIGHT/UP/DOWN, which
    regenerates Section V-G from a cued sample instead of 64 opportunistic
    frames, together with the distribution of the depth cue delta_s that drives
    it.

Every figure is tagged platform-specific or structural, since the distinction
governs what a reader should carry to other hardware.

    python -m tello_gesture_py.scripts.analyse_operational_gesture
    python -m tello_gesture_py.scripts.analyse_operational_gesture --run outputs/runs/<dir>
"""

from tello_gesture_py.src.gestures.evidence import paper_glob
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from tello_gesture_py.src.gestures.build_paper_figures import (
    COL1, COL2, INK, MUTED, C1, C2, plt, tidy)

CLASS_NAMES = {0: "CENTER", 1: "LEFT", 2: "RIGHT", 3: "UP", 4: "DOWN",
               5: "FORWARD", 6: "BACK"}
SHORT = ["CEN", "LEF", "RIG", "UP", "DWN", "FWD", "BCK"]
DEPTH = {"FORWARD", "BACK"}
LATERAL = {"LEFT", "RIGHT", "UP", "DOWN"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def load(runs):
    frames = []
    for r in runs:
        p = os.path.join(r, "op_gesture.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            if len(d):
                d["run"] = os.path.basename(os.path.normpath(r))
                frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    return d[d["phase"] == "hold"].reset_index(drop=True)


def primary(name):
    """The lateral component of a rule label; the whole label if there is none.

    RuleBasedGesture emits compound labels (DOWN-FORWARD). The commanded axis is
    the lateral part, and the depth part is the contamination Section V-G
    measures, so the two are scored separately rather than as one wrong class.
    """
    if not isinstance(name, str) or not name:
        return None
    parts = name.split("-")
    for p in parts:
        if p in LATERAL:
            return p
    return parts[0]


def score(d, pred_col, label):
    """Accuracy, detection rate and confusion for one classifier column."""
    n = len(d)
    det = d["hand_detected"].astype(int) == 1
    n_det = int(det.sum())

    is_rule = pred_col.startswith("rule")
    pred = d[pred_col].apply(primary) if is_rule else d[pred_col]
    truth = d["cued_name"]
    correct = det & (pred == truth)
    k = int(correct.sum())
    k_det = int((correct & det).sum())

    lo, hi = wilson(k, n)
    lod, hid = wilson(k_det, n_det) if n_det else (0.0, 0.0)
    lodet, hidet = wilson(n_det, n)

    cm = np.zeros((7, 8), dtype=int)
    name_to_id = {v: k2 for k2, v in CLASS_NAMES.items()}
    for _, r in d.iterrows():
        t = name_to_id.get(r["cued_name"])
        if t is None:
            continue
        if not (r["hand_detected"] == 1):
            cm[t, 7] += 1
            continue
        p = primary(r[pred_col]) if is_rule else r[pred_col]
        j = name_to_id.get(p)
        cm[t, j if j is not None else 7] += 1

    per_class = {}
    for i in range(7):
        nm = CLASS_NAMES[i]
        tp = int(cm[i, i])
        fn = int(cm[i].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        g = d[d["cued_name"] == nm]
        gd = int((g["hand_detected"].astype(int) == 1).sum())
        per_class[nm] = {"n": len(g), "detection_rate": round(gd / len(g), 4) if len(g) else None,
                         "precision": round(prec, 4), "recall": round(rec, 4),
                         "f1": round(f1, 4)}

    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

    # Hold-level rate. One hold is one independent observation: the operator was
    # cued once and formed the gesture once. Its frames are near-duplicates, so
    # the frame-level interval below is optimistic and this one is the honest
    # denominator. A hold counts correct when the majority of its detected
    # frames carry the cued class.
    # Group by (round, cued class), which identifies a hold regardless of how
    # the run was segmented. Captures written before the hold index was made
    # resume-stable would otherwise merge distinct holds.
    holds, hold_ok, hold_nodet = 0, 0, 0
    for _, g in d.groupby(["round", "cued_label"]):
        holds += 1
        gd = g[g["hand_detected"].astype(int) == 1]
        if not len(gd):
            hold_nodet += 1
            continue
        gp = gd[pred_col].apply(primary) if is_rule else gd[pred_col]
        vote = gp.mode()
        if len(vote) and vote.iloc[0] == g["cued_name"].iloc[0]:
            hold_ok += 1
    lo_h, hi_h = wilson(hold_ok, holds) if holds else (0.0, 0.0)

    return {
        "classifier": label, "frames": n,
        "holds": holds,
        "hold_accuracy": round(hold_ok / holds, 4) if holds else None,
        "hold_accuracy_ci95": [round(lo_h, 4), round(hi_h, 4)],
        "holds_with_no_detection": hold_nodet,
        "detection_rate": round(n_det / n, 4),
        "detection_rate_ci95": [round(lodet, 4), round(hidet, 4)],
        "accuracy_all_frames": round(k / n, 4),
        "accuracy_all_frames_ci95": [round(lo, 4), round(hi, 4)],
        "accuracy_given_detected": round(k_det / n_det, 4) if n_det else None,
        "accuracy_given_detected_ci95": [round(lod, 4), round(hid, 4)],
        "detection_failures": n - n_det,
        "misclassifications": n_det - k_det,
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix_with_nodetect": cm.tolist(),
    }


def depth_contamination(d):
    """Spurious FORWARD/BACK on frames cued to a lateral class (Section V-G)."""
    g = d[d["cued_name"].isin(LATERAL) & (d["hand_detected"].astype(int) == 1)]
    g = g[g["rule_name"].notna()]
    n = len(g)

    def parts(x):
        return set(str(x).split("-"))

    # Two distinct outcomes, kept apart. A compound label (DOWN-FORWARD) sends
    # the commanded axis *plus* an uncommanded depth velocity, which is what
    # Section V-G measures. A bare depth label on a lateral cue is an ordinary
    # misclassification and is already counted in the confusion matrix; pooling
    # them would let classifier error inflate the contamination rate.
    compound = g["rule_name"].apply(lambda x: bool(parts(x) & DEPTH) and bool(parts(x) & LATERAL))
    anydepth = g["rule_name"].apply(lambda x: bool(parts(x) & DEPTH))
    k = int(compound.sum())
    k_any = int(anydepth.sum())
    lo, hi = wilson(k, n) if n else (0.0, 0.0)
    ds = pd.to_numeric(g["delta_s"], errors="coerce").dropna()
    return {
        "cued_lateral_frames": n, "contaminated": k,
        "pct": round(100 * k / n, 2) if n else None,
        "ci95_pct": [round(100 * lo, 2), round(100 * hi, 2)],
        "any_depth_component": k_any,
        "any_depth_pct": round(100 * k_any / n, 2) if n else None,
        "delta_s_abs_median": round(float(ds.abs().median()), 4) if len(ds) else None,
        "delta_s_abs_p95": round(float(np.percentile(ds.abs(), 95)), 4) if len(ds) else None,
        "scale_threshold": 0.18,
        "delta_s_over_threshold_pct": round(100 * float((ds.abs() > 0.18).mean()), 2) if len(ds) else None,
    }


def depth_events(d, col="rule_name"):
    """Depth classes scored per hold, not per frame.

    FORWARD and BACK are motion-defined: only the thrust frames carry the label,
    so a frame-level rate for them measures how much of the window was spent
    moving rather than whether the system recognised the gesture. The event
    question is whether the cued component appeared at all during the hold, and
    whether its opposite appeared too.
    """
    out = {}
    for cls in sorted(DEPTH):
        g = d[d["cued_name"] == cls]
        if g.empty:
            continue
        holds = hit = opp = 0
        frames_with = 0
        other = (DEPTH - {cls}).pop()
        for _, h in g.groupby(["round", "cued_label"]):
            holds += 1
            lab = h[col].dropna().astype(str)
            parts = lab.apply(lambda x: set(x.split("-")))
            has = parts.apply(lambda p: cls in p)
            frames_with += int(has.sum())
            if has.any():
                hit += 1
            if parts.apply(lambda p: other in p).any():
                opp += 1
        out[cls] = {
            "holds": holds,
            "holds_with_cued_component": hit,
            "hold_hit_rate": round(hit / holds, 3) if holds else None,
            "holds_also_showing_opposite": opp,
            "opposite_rate": round(opp / holds, 3) if holds else None,
            "frames_carrying_cued_component": frames_with,
            "frames_total": len(g),
            "frame_share": round(frames_with / len(g), 3) if len(g) else None,
        }
    return out


def fig_operational(res, outdir):
    """Confusion matrices in the Figure 1 idiom, one panel per classifier."""
    panels = [r for r in res if r is not None]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(COL1 * len(panels) * 1.15, 2.9), squeeze=False)
    labels = SHORT + ["none"]
    for col, r in enumerate(panels):
        ax = axes[0][col]
        cm = np.array(r["confusion_matrix_with_nodetect"], dtype=float)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(8))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(7))
        ax.set_yticklabels(SHORT if col == 0 else [], fontsize=6)
        ax.set_title("(%s) %s, operational   %.3f"
                     % ("ab"[col], r["classifier"], r["accuracy_all_frames"]), fontsize=7)
        ax.set_xlabel("predicted", fontsize=7)
        if col == 0:
            ax.set_ylabel("cued", fontsize=7)
        for i in range(7):
            for j in range(8):
                v = cmn[i, j]
                if v >= 0.01:
                    ax.text(j, i, ("%.2f" % v).lstrip("0"), ha="center", va="center",
                            fontsize=4.8, color="white" if v > 0.55 else INK)
        # The no-detection column is a different kind of outcome from the seven
        # class columns, so it is fenced off rather than read as an eighth class.
        ax.axvline(6.5, color=MUTED, linewidth=0.8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.016, pad=0.012)
    cb.set_label("row-normalised rate", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)
    p = os.path.join(outdir, "fig_operational.pdf")
    fig.savefig(p)
    plt.close(fig)
    return p


def tex_table(res, depth, out):
    lines = [r"\begin{table}[t]",
             r"\caption{Gesture accuracy $\uparrow$ on the ground through the "
             r"aircraft video pipeline. Per-hold scores each hold by majority "
             r"vote.}",
             r"\label{tab:operational}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.02}",
             r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Classifier & Frames & Detected & Accuracy & Acc.\ $\mid$ det. & Per hold \\",
             r"\midrule"]
    for r in res:
        if r is None:
            continue
        lines.append("%s & %d & %.3f & %.3f & %.3f & %.3f \\\\" % (
            r["classifier"], r["frames"], r["detection_rate"],
            r["accuracy_all_frames"],
            r["accuracy_given_detected"] if r["accuracy_given_detected"] is not None else float("nan"),
            r["hold_accuracy"] if r.get("hold_accuracy") is not None else float("nan")))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--exclude", action="append", default=None,
                    metavar="CLASS:ROUNDS",
                    help="Declared operator error, e.g. LEFT:0-2. Those holds are "
                         "dropped and the decision is recorded in the output. "
                         "Exclusion rather than relabelling: correcting the cue "
                         "from the classifier's own output would be circular.")
    ap.add_argument("--out_json", default="outputs/metrics/operational_gesture.json")
    ap.add_argument("--figdir", default="paper/figures")
    ap.add_argument("--tabdir", default="paper/tables")
    args = ap.parse_args()

    # default: the paper's evidence set; pass --run for newer captures
    runs = args.run or sorted(paper_glob("outputs/runs/*/"))
    d = load(runs)
    if d.empty:
        raise SystemExit("No op_gesture.csv hold frames found. "
                         "Run scripts/operational_gesture_eval.py first.")

    excluded = {}
    if args.exclude:
        mask = pd.Series(False, index=d.index)
        for spec in args.exclude:
            cls, _, rng = spec.partition(":")
            lo, _, hi = rng.partition("-")
            lo = int(lo)
            hi = int(hi) if hi else lo
            m = (d["cued_name"] == cls) & (d["round"] >= lo) & (d["round"] <= hi)
            excluded[spec] = {"frames": int(m.sum()),
                              "holds": int(d[m].groupby(["round", "cued_label"]).ngroups)}
            mask |= m
        n_before = len(d)
        d = d[~mask].reset_index(drop=True)
        print("declared operator-error exclusions:")
        for k, v in excluded.items():
            print(f"   {k:<12} {v['holds']} holds, {v['frames']} frames")
        print(f"   {n_before} -> {len(d)} frames\n")

    res = [score(d, "rule_name", "Rule-based (deployed path)")]
    if "rule_raw" in d.columns and d["rule_raw"].notna().any():
        res.append(score(d, "rule_raw", "Rule-based (ungated)"))
    if d["svm_name"].notna().any():
        res.insert(0, score(d, "svm_name", "RBF-SVM"))
    else:
        print("NOTE: no SVM columns in this capture; only the rule path is scored.\n")

    dep = depth_contamination(d)

    for r in res:
        print(f"=== {r['classifier']}  ({r['frames']} cued frames) ===")
        print(f"  hand detected          {r['detection_rate']:.4f}  "
              f"[{r['detection_rate_ci95'][0]:.3f}, {r['detection_rate_ci95'][1]:.3f}]")
        print(f"  accuracy, all frames   {r['accuracy_all_frames']:.4f}  "
              f"[{r['accuracy_all_frames_ci95'][0]:.3f}, {r['accuracy_all_frames_ci95'][1]:.3f}]")
        ag = r["accuracy_given_detected"]
        print(f"  accuracy | detected    {ag:.4f}  "
              f"[{r['accuracy_given_detected_ci95'][0]:.3f}, "
              f"{r['accuracy_given_detected_ci95'][1]:.3f}]" if ag is not None else "  n/a")
        print(f"  detection failures     {r['detection_failures']}")
        print(f"  misclassifications     {r['misclassifications']}")
        print(f"  macro F1               {r['macro_f1']:.4f}")
        if r.get("holds"):
            print(f"  hold-level accuracy    {r['hold_accuracy']:.4f}  "
                  f"[{r['hold_accuracy_ci95'][0]:.3f}, {r['hold_accuracy_ci95'][1]:.3f}]"
                  f"   over {r['holds']} holds  <- independent unit")
            if r["holds_with_no_detection"]:
                print(f"  holds with no detection {r['holds_with_no_detection']}")
        print(f"  {'class':<9}{'n':>6}{'detect':>9}{'F1':>8}")
        for k, v in r["per_class"].items():
            print(f"  {k:<9}{v['n']:>6}{v['detection_rate']:>9.3f}{v['f1']:>8.3f}")
        print()

    ev = depth_events(d)
    if ev:
        print("=== depth classes scored as events (motion-defined, see note) ===")
        print(f"  {'cued':<9}{'holds':>7}{'hit':>6}{'hit rate':>10}"
              f"{'opposite':>10}{'frame share':>13}")
        for k, v in ev.items():
            print(f"  {k:<9}{v['holds']:>7}{v['holds_with_cued_component']:>6}"
                  f"{v['hold_hit_rate']:>10.3f}{v['opposite_rate']:>10.3f}"
                  f"{v['frame_share']:>13.3f}")
        print()
    out_depth_events = ev

    print("=== depth-cue contamination (Section V-G, cued sample) ===")
    print(f"  cued lateral frames    {dep['cued_lateral_frames']}")
    print(f"  compound contamination {dep['contaminated']}  ({dep['pct']}%)  "
          f"[{dep['ci95_pct'][0]}, {dep['ci95_pct'][1]}]   <- Section V-G")
    print(f"  any depth component    {dep['any_depth_component']}  ({dep['any_depth_pct']}%)"
          f"   (includes outright misclassification)")
    print(f"  |delta_s| median       {dep['delta_s_abs_median']}")
    print(f"  |delta_s| p95          {dep['delta_s_abs_p95']}  (threshold 0.18)")
    print(f"  frames over threshold  {dep['delta_s_over_threshold_pct']}%")

    provenance = {
        "platform_specific": [
            "detection_rate -- depends on this camera, its H.264 encoder and the link rate",
            "accuracy_all_frames -- inherits the detection rate, so it moves with the platform",
            "delta_s magnitude -- scales with the frame interval this link delivers",
        ],
        "structural": [
            "hold_accuracy vs accuracy_all_frames -- the gap between the independent "
            "unit and the frame count is the same near-duplicate effect the paper "
            "identifies in within-session splitting, reproduced in its own measurement",
            "accuracy_given_detected vs the offline within-session figure -- the gap is the "
            "evaluation protocol, not the hardware",
            "spurious FORWARD/BACK on cued lateral frames -- delta_s is differenced between "
            "frames without normalising by delta_t, so it is wrong wherever the rate varies",
            "rule-vs-SVM disagreement on identical frames -- a property of the two classifiers",
        ],
    }

    out = {"runs": sorted(d["run"].unique().tolist()), "hold_frames": len(d),
           "declared_exclusions": excluded,
           "classifiers": res, "depth_contamination": dep,
           "depth_events": out_depth_events,
           "result_provenance": provenance}

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out_json}")

    os.makedirs(args.figdir, exist_ok=True)
    p = fig_operational(res, args.figdir)
    if p:
        print(f"wrote {p}")
    os.makedirs(args.tabdir, exist_ok=True)
    print("wrote " + tex_table(res, dep, os.path.join(args.tabdir, "tab_operational.tex")))

    print("\nplatform-specific:")
    for s in provenance["platform_specific"]:
        print("  -", s)
    print("structural:")
    for s in provenance["structural"]:
        print("  -", s)


if __name__ == "__main__":
    main()
