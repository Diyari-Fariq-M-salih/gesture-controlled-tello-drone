"""Summarize a run's perf log into the Table A "Runtime Performance" numbers.

Reports FPS, per-stage latency, camera-to-command latency, RC command rate and
LLM explanation latency.

Two things this deliberately does NOT do:

* It does not average a throttled stage over frames where it did not run.
  Stage columns are only populated on frames where the stage executed, so the
  mean here is the cost of an actual detection. The proportion of frames on
  which each stage ran is reported separately as its duty cycle -- that pair
  (conditional latency + duty cycle) is what belongs in the paper, not a single
  diluted average.

* It does not silently trust `fps_inst`. The controller logs one row per
  distinct decoded frame, so the rate is computed from the wall-clock span of
  the run rather than from per-row reciprocals, which are dominated by
  scheduling noise at the tails.

Usage:
    python -m tello_gesture_py.src.gestures.summarize_perf --run outputs/runs/<run-dir>
    python -m tello_gesture_py.src.gestures.summarize_perf --perf_csv <path/to/perf.csv>
    ... add --markdown to emit a LaTeX-pasteable table.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = [
    ("hand_detect_ms", "Hand detection (MediaPipe Hands)"),
    ("face_detect_ms", "Face detection (MediaPipe)"),
    ("face_id_ms", "Face embedding + cosine match"),
    ("svm_infer_ms", "Gesture SVM inference"),
    ("rule_infer_ms", "Rule-based gesture inference"),
    ("mode_manager_ms", "Mode-manager update"),
    ("command_exec_ms", "Command execution"),
    ("camera_to_command_ms", "End-to-end camera-to-command"),
    ("frame_age_ms", "Frame age at consumption (decode + queue)"),
]


def _stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p95": float(np.percentile(s, 95)),
        "p99": float(np.percentile(s, 99)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def _fmt(st: dict, unit: str = " ms") -> str:
    if not st.get("n"):
        return "n/a"
    return (f"mean {st['mean']:.2f}{unit}   median {st['median']:.2f}{unit}   "
            f"p95 {st['p95']:.2f}{unit}   n={st['n']}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="Run directory under outputs/runs/")
    g.add_argument("--perf_csv", help="Path to a perf.csv written by PerfLogger")
    ap.add_argument("--markdown", action="store_true", help="Emit a markdown table too")
    args = ap.parse_args()

    if args.run:
        run_dir = Path(args.run)
        perf_path = run_dir / "perf.csv"
    else:
        perf_path = Path(args.perf_csv)
        run_dir = perf_path.parent

    events_path = perf_path.with_name(perf_path.stem + "_events.csv")
    manifest_path = run_dir / "manifest.json"

    df = pd.read_csv(perf_path)
    print(f"Run:    {run_dir.name}")
    print(f"Frames: {len(df)}")

    if manifest_path.exists():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        abl = man.get("ablations") or ["none"]
        print(f"Ablations: {', '.join(abl)}")
        if man.get("note"):
            print(f"Note:   {man['note']}")
        summ = man.get("summary", {})
        if summ.get("duplicate_frames_skipped") is not None:
            print(f"Duplicate frame observations skipped: {summ['duplicate_frames_skipped']}")

    # ---- frame rate from wall-clock span, not per-row reciprocals ----
    print("\n--- Table A: Runtime Performance ---\n")
    span = float(df["t"].max() - df["t"].min()) if len(df) > 1 else 0.0
    if span > 0:
        print(f"{'Frame rate (run mean):':<46} {(len(df) - 1) / span:.2f} fps over {span:.1f} s")
    dt = _stats(df["dt_s"] * 1000.0) if "dt_s" in df else {"n": 0}
    if dt.get("n"):
        print(f"{'Inter-frame interval:':<46} {_fmt(dt)}")

    print()
    rows = []
    for col, label in STAGES:
        if col not in df.columns:
            continue
        st = _stats(df[col])
        if not st.get("n"):
            continue
        ran_col = f"{col}_ran"
        duty = ""
        if ran_col in df.columns:
            ran = pd.to_numeric(df[ran_col], errors="coerce").fillna(0)
            duty = f"  [ran on {int(ran.sum())}/{len(df)} frames = {100.0 * ran.mean():.0f}%]"
        print(f"{label + ':':<46} {_fmt(st)}{duty}")
        rows.append((label, st))

    # ---- event rates ----
    print()
    if events_path.exists():
        ev = pd.read_csv(events_path)

        rc = ev[ev["event"] == "rc_send"]["t"].values
        if len(rc) > 1:
            print(f"{'RC command output rate:':<46} {(len(rc) - 1) / (rc[-1] - rc[0]):.2f} Hz  (n={len(rc)})")
            jitter = _stats(pd.Series(np.diff(rc) * 1000.0))
            print(f"{'RC send interval:':<46} {_fmt(jitter)}")
        else:
            print(f"{'RC command output rate:':<46} n/a (need >=2 sends)")

        llm = ev[ev["event"] == "llm_explanation"]
        if not llm.empty:
            print(f"{'LLM explanation latency:':<46} {_fmt(_stats(llm['latency_ms']))}")
        else:
            print(f"{'LLM explanation latency:':<46} n/a (none logged)")

        for name, label in (("battery_failsafe_land", "Battery failsafe landings"),
                            ("emergency_stop", "Emergency stops")):
            k = int((ev["event"] == name).sum())
            if k:
                print(f"{label + ':':<46} {k}")
    else:
        print(f"(no events file at {events_path})")

    if args.markdown and rows:
        print("\n--- markdown ---\n")
        print("| Stage | Mean (ms) | Median (ms) | p95 (ms) | n |")
        print("|---|---:|---:|---:|---:|")
        for label, st in rows:
            print(f"| {label} | {st['mean']:.2f} | {st['median']:.2f} | {st['p95']:.2f} | {st['n']} |")


if __name__ == "__main__":
    main()
