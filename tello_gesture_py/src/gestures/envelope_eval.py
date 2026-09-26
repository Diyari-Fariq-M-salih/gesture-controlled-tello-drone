"""Authorization envelope and on-ground false rejection, at flight-free n.

Figure 2's envelope rests on bins holding two and three frames, and it carries a
concrete design recommendation (the face-following standoff). The on-ground row
of the identity table rests on 122 frames and reports a rate worse than flight,
an inversion the paper does not explain. Both quantities are capturable on the
ground at much higher n, which is what `scripts/stream_capture.py --mode
identity` produces.

This script consumes those captures and reports:

  * authorization against apparent face size, in the same bins as Figure 2, with
    Wilson intervals and the frame count per bin;
  * the same, against the operator's declared standoff distance, which is the
    quantity a reader would actually act on;
  * the on-ground false rejection rate, split by whether the face was detected
    at all, since a frame with no detection is not a rejection by the embedding.

That last split is what makes the ground-versus-flight inversion legible: on the
ground the operator can stand close enough to clip the padded crop, a regime
flight rarely visits.

    python -m tello_gesture_py.src.gestures.envelope_eval
"""

from tello_gesture_py.src.gestures.evidence import paper_glob
import argparse
import glob
import json
import math
import os

import pandas as pd

BINS = [0, 150, 180, 220, 280, 350, 480, 620]


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
        p = os.path.join(r, "stream_identity.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            if len(d):
                d["run"] = os.path.basename(os.path.normpath(r))
                frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _rate_table(d, group_col, groups, label):
    rows = []
    for key, g in groups:
        if not len(g):
            continue
        k, n = int(g["face_auth"].sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({label: key, "n": n, "auth_pct": round(100 * k / n, 1),
                     "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
                     "median_score": round(float(g["faceid_score"].median()), 3)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None)
    ap.add_argument("--out_json", default="outputs/metrics/envelope_eval.json")
    args = ap.parse_args()

    # default: the paper's evidence set; pass --run for newer captures
    runs = args.run or sorted(paper_glob("outputs/runs/*/"))
    d = load(runs)
    if d.empty:
        raise SystemExit("No stream_identity.csv found. Run scripts/stream_capture.py --mode identity first.")

    total = len(d)
    detected = d[d["face_raw"].astype(int) == 1]
    scored = detected[detected["faceid_score"].notna()]

    by_size = _rate_table(
        scored, "face_bbox_h",
        [(f"{lo}-{hi}", scored[(scored["face_bbox_h"] >= lo) & (scored["face_bbox_h"] < hi)])
         for lo, hi in zip(BINS[:-1], BINS[1:])],
        "bbox_h_px")

    by_dist = _rate_table(
        scored, "distance_m",
        [(float(v), scored[scored["distance_m"] == v]) for v in sorted(scored["distance_m"].dropna().unique())],
        "distance_m")

    k = int(scored["face_auth"].sum())
    lo, hi = wilson(k, len(scored))
    out = {
        "runs": sorted(d["run"].unique().tolist()),
        "frames_total": total,
        "frames_face_detected": len(detected),
        "frames_scored": len(scored),
        "detection_rate": round(len(detected) / total, 4),
        "frr_pct_given_scored": round(100 * (1 - k / len(scored)), 2),
        "frr_ci95_pct": [round(100 * (1 - hi), 2), round(100 * (1 - lo), 2)],
        "by_bbox_height": by_size,
        "by_distance": by_dist,
    }

    print(f"frames {total}   face detected {len(detected)} "
          f"({100 * len(detected) / total:.1f}%)   scored {len(scored)}")
    print(f"on-ground FRR (scored frames) {out['frr_pct_given_scored']:.2f}%  "
          f"[{out['frr_ci95_pct'][0]}, {out['frr_ci95_pct'][1]}]\n")

    print(f"{'bbox h (px)':<14}{'n':>7}{'auth %':>9}{'95% CI':>18}{'med score':>11}")
    for r in by_size:
        print(f"{r['bbox_h_px']:<14}{r['n']:>7}{r['auth_pct']:>9.1f}"
              f"{str(r['ci95']):>18}{r['median_score']:>11.3f}")

    if by_dist:
        print(f"\n{'distance (m)':<14}{'n':>7}{'auth %':>9}{'95% CI':>18}{'med score':>11}")
        for r in by_dist:
            print(f"{r['distance_m']:<14.1f}{r['n']:>7}{r['auth_pct']:>9.1f}"
                  f"{str(r['ci95']):>18}{r['median_score']:>11.3f}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
