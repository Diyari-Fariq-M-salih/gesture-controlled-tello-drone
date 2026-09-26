"""Calibrate hand-face association on the operator's own capture frames.

The dataset frames in data/raw/ are unflipped webcam captures of one operator
forming each gesture, which makes them a ground truth for two questions the
associator's settings depend on:

  1. Which (target_side, mirrored) setting accepts the operator's gesturing
     hand? Exactly one anatomically consistent setting should; its mirror
     image scores the same because the two flips cancel.
  2. How far is the gesturing hand from its own arm, and from the other arm?
     That fixes dist_thresh and shows what side_margin costs.

Faces come from the deployed FaceFollower, hands from HandGesture, skeletons
from the real pose model. Frames without a detected face, hand or matching
skeleton are counted and skipped.

    python -m tello_gesture_py.scripts.association_calibration
    python -m tello_gesture_py.scripts.association_calibration --per-class 30 --seed 3

Writes outputs/metrics/association_calibration.json.
"""
import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict

import cv2
import numpy as np

from tello_gesture_py.src.face_follow import FaceFollower
from tello_gesture_py.src.hand_association import HandAssociator, _candidates
from tello_gesture_py.src.hand_gesture import HandGesture

SESSION_ROOTS = {"A": "data/raw", "B": "data/raw/sessionB",
                 "C": "data/raw/sessionC", "D": "data/raw/sessionD"}
SETTINGS = [(s, m) for s in ("right", "left") for m in (False, True)]
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 1.0)


def _q(a, p):
    return round(float(np.percentile(a, p)), 3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=15,
                    help="frames sampled per gesture class per session")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="outputs/metrics/association_calibration.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    assoc = HandAssociator()
    hands = HandGesture(max_num_hands=4)
    stage = Counter()
    accepted = {k: 0 for k in SETTINGS}
    frames = []   # per evaluable frame: class, and per-setting (d_target, d_other) of best hand

    for sess, root in SESSION_ROOTS.items():
        for cdir in sorted(glob.glob(os.path.join(root, "0*"))):
            cls = os.path.basename(cdir)[3:]
            files = sorted(glob.glob(os.path.join(cdir, "*.jpg")))
            for f in rng.sample(files, min(args.per_class, len(files))):
                frame = cv2.imread(f)
                if frame is None:
                    continue
                stage["frames"] += 1
                face = FaceFollower()
                face.cfg.detect_every_n = 1
                face.observe(frame)
                bbox = face.get_last_bbox() if face.face_detected() else None
                det = hands.detect(frame)
                cands = _candidates(det)
                if bbox is None:
                    stage["no_face"] += 1
                    continue
                if not cands:
                    stage["no_hand"] += 1
                    continue
                pose = assoc.detect(frame)
                skel, _ = assoc.find_skeleton(pose, bbox, frame.shape)
                if skel is None:
                    stage["no_skeleton"] += 1
                    continue
                stage["evaluable"] += 1
                pl = pose.pose_landmarks[skel]
                scale = assoc._pose_scale(pl, bbox, frame.shape)
                row = {"session": sess, "class": cls}
                for side, mir in SETTINGS:
                    assoc.target_side, assoc.mirrored = side, mir
                    if any(assoc.score_candidate(lm, pl, scale, lab, sc)[2]
                           for lm, lab, sc in cands):
                        accepted[(side, mir)] += 1
                    ps, po, _ = assoc._resolve_sides()
                    row[f"{side}_{int(mir)}"] = min(
                        ((assoc.hand_against_skeleton(lm, pl, ps, scale),
                          assoc.hand_against_skeleton(lm, pl, po, scale))
                         for lm, _, _ in cands), key=lambda t: t[0])
                frames.append(row)

    n = stage["evaluable"]
    if n == 0:
        print("no evaluable frames; is data/raw/ present?")
        return 1
    rates = {f"{s}_mirrored={m}": round(accepted[(s, m)] / n, 3) for s, m in SETTINGS}
    best_side, best_mir = max(SETTINGS, key=lambda k: (accepted[k], not k[1]))
    key = f"{best_side}_{int(best_mir)}"
    dt = np.array([r[key][0] for r in frames])
    do = np.array([r[key][1] for r in frames])
    per_class = defaultdict(list)
    for r in frames:
        per_class[r["class"]].append(r[key][0])

    out = {
        "frames": dict(stage),
        "acceptance_at_default_thresholds": rates,
        # The two cancelling settings always tie; data/raw frames are unflipped,
        # so the unmirrored one of the pair is the anatomically correct reading.
        "setting_for_unflipped_frames": {"target_side": best_side, "mirrored": best_mir},
        "d_target": {"p50": _q(dt, 50), "p90": _q(dt, 90), "p95": _q(dt, 95),
                     "p99": _q(dt, 99), "max": round(float(dt.max()), 3)},
        "d_other": {"p1": _q(do, 1), "p5": _q(do, 5), "p50": _q(do, 50),
                    "min": round(float(do.min()), 3)},
        "acceptance_by_dist_thresh": {
            str(th): round(float(np.mean((dt <= th) & (dt < 0.85 * do))), 3)
            for th in THRESHOLDS},
        "per_class_d_target": {
            c: {"n": len(v), "p50": _q(v, 50), "p90": _q(v, 90)}
            for c, v in sorted(per_class.items())},
        "params": {"per_class": args.per_class, "seed": args.seed,
                   "side_margin": 0.85, "pose_model": assoc.model_path},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"frames: {dict(stage)}")
    for k, v in rates.items():
        print(f"  {k:26s} accepts {100 * v:4.0f}%")
    print(f"setting for these (unflipped) frames: target_side={best_side} mirrored={best_mir}")
    print(f"d_target p50={out['d_target']['p50']} p95={out['d_target']['p95']}   "
          f"d_other p5={out['d_other']['p5']}")
    for th, v in out["acceptance_by_dist_thresh"].items():
        print(f"  dist_thresh={th}: accepts {100 * v:.0f}%")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
