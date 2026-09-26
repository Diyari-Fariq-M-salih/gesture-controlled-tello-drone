"""Score a free flight against an independent video annotation.

The cued protocol asks the operator to follow a schedule. This does the reverse:
the operator flies naturally, an external recording captures what they actually
did, and the annotation of that recording supplies the ground truth. The
classifier's own output plays no part in deciding what was intended, which is
what makes the comparison independent.

Two clocks have to be reconciled. The flight log runs on the host's wall clock;
the phone recording runs on its own. A sync marker -- key 'm' in the controller,
a clap at the same instant -- gives one shared event, and every annotated segment
is expressed as an offset from it.

Procedure:

  1. Fly with --freeflight-log. Press 'm' and clap at the start, and again at the
     end if the flight is long.
  2. Annotate the phone video into a CSV *before opening the flight log*. The
     order matters: knowing what the drone did would contaminate a judgement
     about what you meant.
  3. Run this.

Annotation CSV (seconds measured from the clap):

    start_s,end_s,intended
    4.2,6.0,LEFT
    9.1,11.4,FORWARD
    15.0,17.2,NONE          <- deliberately no gesture; scores false positives

    python -m tello_gesture_py.scripts.score_against_video \\
        --run outputs/runs/<dir> --annotations annotations.csv
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

LATERAL = {"LEFT", "RIGHT", "UP", "DOWN"}
DEPTH = {"FORWARD", "BACK"}
CLASSES = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN", "FORWARD", "BACK"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def components(label):
    if not isinstance(label, str) or not label or label == "NOHAND":
        return set()
    return {p for p in label.split("-") if p in LATERAL | DEPTH | {"CENTER"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--marker", type=int, default=1,
                    help="Which sync marker the annotation offsets are measured from.")
    ap.add_argument("--out_json", default="outputs/metrics/video_scored.json")
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.run, "manifest.json"), encoding="utf-8"))
    markers = man.get("sync_markers") or []
    if len(markers) < args.marker:
        raise SystemExit(
            f"run has {len(markers)} sync markers; need at least {args.marker}. "
            "Press 'm' in flight and clap at the same moment.")
    t0 = markers[args.marker - 1]

    dense = os.path.join(args.run, "cued.csv")
    if not os.path.exists(dense):
        raise SystemExit("no dense log; fly with --freeflight-log")
    d = pd.read_csv(dense)
    d = d[d["phase"] == "freeflight"] if "phase" in d else d
    if d.empty:
        raise SystemExit("dense log holds no free-flight frames")

    ann = list(csv.DictReader(open(args.annotations, encoding="utf-8")))
    if not ann:
        raise SystemExit("annotation file is empty")

    rows, cm = [], np.zeros((len(CLASSES) + 1, len(CLASSES) + 1), dtype=int)
    idx = {c: i for i, c in enumerate(CLASSES)}
    idx["NONE"] = len(CLASSES)

    for a in ann:
        s, e = float(a["start_s"]), float(a["end_s"])
        want = a["intended"].strip().upper()
        w = d[(d["t"] >= t0 + s) & (d["t"] <= t0 + e)]
        if w.empty:
            rows.append({"intended": want, "frames": 0, "verdict": "no_frames",
                         "start_s": s, "end_s": e})
            continue

        emitted = w["emitted_gesture"].dropna().astype(str)
        det = int((w["hand_detected"].astype(int) == 1).sum())
        comps = [components(x) for x in emitted]
        carried = sum(1 for c in comps if want in c) if want != "NONE" else 0
        any_cmd = sum(1 for c in comps if c - {"CENTER"})

        if want == "NONE":
            verdict = "clean" if any_cmd == 0 else "false_positive"
            got = "NONE" if any_cmd == 0 else "CMD"
        else:
            verdict = "hit" if carried > 0 else "miss"
            # the component the segment most often carried, for the matrix
            tally = {}
            for c in comps:
                for k in c:
                    tally[k] = tally.get(k, 0) + 1
            got = max(tally, key=tally.get) if tally else "NONE"

        extra = sorted({k for c in comps for k in c} - {want, "CENTER"})
        rows.append({
            "intended": want, "start_s": s, "end_s": e,
            "frames": len(w), "detected": det,
            "frames_carrying_intended": carried,
            "share": round(carried / len(w), 3),
            "dominant_emitted": got, "extra_components": extra,
            "verdict": verdict,
        })
        if want in idx and got in idx:
            cm[idx[want], idx[got]] += 1

    out = pd.DataFrame(rows)
    scored = out[out["verdict"].isin(["hit", "miss"])]
    none_seg = out[out["verdict"].isin(["clean", "false_positive"])]

    print(f"run        {os.path.basename(os.path.normpath(args.run))}")
    print(f"classifier {man.get('classifier')}")
    print(f"segments   {len(out)}  ({len(scored)} commanded, {len(none_seg)} idle)\n")

    if len(scored):
        k = int((scored["verdict"] == "hit").sum())
        lo, hi = wilson(k, len(scored))
        print(f"segment-level: the intended command reached the aircraft in "
              f"{k}/{len(scored)} = {k/len(scored):.3f}  [{lo:.3f}, {hi:.3f}]")
        print(f"median share of a segment carrying it: "
              f"{scored['share'].median():.3f}\n")
        print(f"{'intended':<9}{'n':>4}{'hit':>5}{'rate':>7}{'median share':>14}")
        for c in CLASSES:
            g = scored[scored["intended"] == c]
            if not len(g):
                continue
            h = int((g["verdict"] == "hit").sum())
            print(f"{c:<9}{len(g):>4}{h:>5}{h/len(g):>7.3f}{g['share'].median():>14.3f}")

    if len(none_seg):
        fp = int((none_seg["verdict"] == "false_positive").sum())
        print(f"\nidle segments emitting a command: {fp}/{len(none_seg)}")

    bad = out[out["verdict"] == "no_frames"]
    if len(bad):
        print(f"\n{len(bad)} annotated segments had no logged frames -- check the "
              "sync marker and the offsets")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    payload = {"run": os.path.basename(os.path.normpath(args.run)),
               "classifier": man.get("classifier"),
               "marker_index": args.marker, "marker_t": t0,
               "segments": rows,
               "confusion_matrix": cm.tolist(),
               "classes": CLASSES + ["NONE"]}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
