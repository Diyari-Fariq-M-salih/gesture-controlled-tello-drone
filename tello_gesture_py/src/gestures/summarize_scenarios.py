"""Summarize a scenario trial log into the "Table B: Scenario-Level System Tests"
markdown table requested by the ARO content review.

Each run writes its trials to outputs/runs/<run>/scenarios.csv (before run
directories existed they went to outputs/metrics/).

Usage:
    python -m tello_gesture_py.src.gestures.summarize_scenarios --scenario_csv outputs/runs/<run>/scenarios.csv
"""

import argparse

import pandas as pd

SCENARIO_LABELS = {
    "authorized_gesture_accepted": "Authorized gesture accepted",
    "unauthorized_gesture_rejected": "Unauthorized gesture rejected",
    "face_following_activates": "Face-following activates correctly",
    "gesture_preempts_face": "Gesture preempts face-following",
    "hysteresis_prevents_flicker": "Hysteresis prevents flicker",
    "search_starts_after_loss": "Search starts after target loss",
    "target_reacquired_after_search": "Target reacquired after search",
    "battery_failsafe_landing": "Battery failsafe landing",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.scenario_csv)

    print("| Test case | Trials | Successes | Success rate |")
    print("|---|---:|---:|---:|")
    for key, label in SCENARIO_LABELS.items():
        sub = df[df["scenario"] == key]
        trials = len(sub)
        successes = int(sub["success"].sum()) if trials else 0
        rate = f"{100.0 * successes / trials:.0f}%" if trials else "TBD"
        trials_str = str(trials) if trials else "TBD"
        successes_str = str(successes) if trials else "TBD"
        print(f"| {label} | {trials_str} | {successes_str} | {rate} |")

    if df.empty:
        print("\n(no trials logged yet)")


if __name__ == "__main__":
    main()
