"""Reveal arm assignments after flying is complete.

Under `--arm-randomise` the controller withholds the arm from the operator for
the duration of the run. This is the sanctioned way to read it back, and it
checks the record while it does so: a manifest that disagrees with the latency
column its own perf log wrote is the failure mode this whole exercise exists to
prevent, so it is reported as an error rather than printed as a fact.

    python -m tello_gesture_py.scripts.reveal_arms
    python -m tello_gesture_py.scripts.reveal_arms --glob "outputs/runs/*_drone_n5-*"
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tello_gesture_py.src import gesture_classifier as gclf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/runs/*/",
                    help="Run directories to reveal.")
    ap.add_argument("--blinded-only", action="store_true",
                    help="Only runs that were flown blinded.")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(args.glob)):
        md = os.path.join(d, "manifest.json")
        if not os.path.exists(md):
            continue
        man = json.load(open(md, encoding="utf-8"))
        arm = man.get("arm_assignment") or {}
        if args.blinded_only and not arm.get("randomised"):
            continue
        v = gclf.verify_run(os.path.normpath(d))
        rows.append({
            "run": os.path.basename(os.path.normpath(d)),
            "declared": v["declared"], "observed": v["observed"],
            "status": v["status"],
            "randomised": bool(arm.get("randomised")),
            "seed": arm.get("seed"),
            "sha": (man.get("classifier_detail") or {}).get("model_sha256"),
            "frozen": (man.get("classifier_detail") or {}).get("model_matches_frozen"),
        })

    if not rows:
        raise SystemExit("no matching runs")

    print(f"{'run':<36}{'arm':>7}{'observed':>10}{'rnd':>5}{'seed':>12}"
          f"{'frozen':>8}  status")
    for r in rows:
        print(f"{r['run']:<36}{str(r['declared']):>7}{str(r['observed']):>10}"
              f"{'y' if r['randomised'] else '-':>5}{str(r['seed']):>12}"
              f"{'y' if r['frozen'] else ('-' if r['frozen'] is None else 'NO'):>8}"
              f"  {r['status']}")

    bad = [r for r in rows if r["status"] in ("MISMATCH", "AMBIGUOUS")]
    counts = {}
    for r in rows:
        if r["observed"]:
            counts[r["observed"]] = counts.get(r["observed"], 0) + 1
    print(f"\nruns with gesture frames by observed arm: {counts}")
    if bad:
        print("\nMANIFEST DISAGREES WITH THE EXECUTED BRANCH:")
        for r in bad:
            print(f"  {r['run']}: declared {r['declared']}, observed {r['observed']}")
        return 1
    print("every manifest agrees with the latency column its run wrote")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
