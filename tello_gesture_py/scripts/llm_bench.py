"""Compare local LLMs for the reason-only explanation, on moments from real flights.

Each decision row of a run records what the controller sent the language model
(mode, command, its own deterministic reason, battery...). This replays a sample
of them through each Ollama model with the controller's exact request
(mode_manager.llm_request) and scores the replies:

    load_s       first call, including loading the model into memory
    median/p95   per-call latency once loaded, ms
    timeouts     calls over --timeout (the controller's is 4 s)
    valid        a JSON reply with a non-empty "reason"
    on_mode      the sentence names what the drone is doing (a keyword of its mode)
    false_bat    mentions the battery while it is above the failsafe
    short        at most 25 words

    python -m tello_gesture_py.scripts.llm_bench --models qwen2.5:0.5b-instruct qwen2.5:1.5b-instruct
    python -m tello_gesture_py.scripts.llm_bench --models llama3.2:3b --n 60 --runs outputs/runs/<run>

Models must be pulled first (ollama pull <name>). Writes a new
outputs/runs/<stamp>_bench_llm/ with every reply, so the scoring can be checked.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tello_gesture_py.src.config import LLMReasonConfig  # noqa: E402
from tello_gesture_py.src.mode_manager import llm_request, parse_reason  # noqa: E402
from tello_gesture_py.src.run_context import RunContext  # noqa: E402

MODE_WORDS = {
    "gesture": r"gestur|hand|command|sign",
    "face": r"face|follow|track|operator|person|user",
    "search_360": r"search|scan|look|rotat|turn|spin|find",
    "hover": r"hover|hold|wait|idle|station|stay|still",
    "land": r"land|battery|descen",
    "follow_back": r"follow|behind|back|trail",
}


def payloads(runs, n, seed):
    """A sample of logged decision rows from the given runs, spread over the modes."""
    frames = [pd.read_csv(f"{r}/decisions.csv", low_memory=False) for r in runs]
    d = pd.concat([f for f in frames if len(f)], ignore_index=True)
    d = d[d["mode"].notna() & d["det_reason"].notna()]
    per = max(1, n // max(d["mode"].nunique(), 1))
    rows = pd.concat([g.sample(min(len(g), per), random_state=seed) for _, g in d.groupby("mode")])
    rows = rows.sample(min(len(rows), n), random_state=seed)
    out = []
    for _, r in rows.iterrows():
        num = lambda v: None if pd.isna(v) else float(v)
        flag = lambda v: False if pd.isna(v) else bool(v)     # older runs lack some columns
        out.append({"mode": r["mode"], "command": r["command"],
                    "deterministic_reason": r["det_reason"],
                    "hand_detected": flag(r.get("hand_auth")), "face_detected": flag(r.get("face_auth")),
                    "battery": num(r["battery"]), "altitude_cm": num(r["altitude_cm"]),
                    "time_since_any_seen_s": num(r.get("t_any")), "flying": flag(r.get("flying"))})
    return out


def ask(cfg, payload, timeout):
    t0 = time.perf_counter()
    try:
        r = requests.post(cfg.url, json=llm_request(cfg, payload), timeout=timeout)
        r.raise_for_status()
        reason, err = parse_reason(r.json()), ""
    except Exception as e:
        reason, err = "", type(e).__name__
    return reason, err, (time.perf_counter() - t0) * 1000.0


def score(payload, reason, err, failsafe=15):
    words = reason.split()
    bat = payload.get("battery")
    return {"timeout": int("Timeout" in err), "valid": int(bool(reason) and not err
                                                          and reason != "LLM: (no reason)"),
            "on_mode": int(bool(re.search(MODE_WORDS.get(payload["mode"], "$^"), reason, re.I))),
            "false_bat": int(bool(re.search(r"batter", reason, re.I)) and bat is not None
                             and bat > failsafe),
            "short": int(0 < len(words) <= 25)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Run directories to sample (default: every flight-stack run).")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--timeout", type=float, default=LLMReasonConfig().timeout_s)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    runs = a.runs or sorted(p for p in glob.glob("outputs/runs/*_drone_*")
                            if os.path.exists(f"{p}/decisions.csv"))
    sample = payloads(runs, a.n, a.seed)
    installed = {m["name"] for m in requests.get(
        LLMReasonConfig().url.replace("/api/chat", "/api/tags"), timeout=5).json()["models"]}
    out = RunContext("bench_llm")
    out.record("llm_bench", {"source_runs": [os.path.basename(r) for r in runs],
                             "n": len(sample), "timeout_s": a.timeout, "models": a.models})
    modes = pd.Series([p["mode"] for p in sample]).value_counts().to_dict()
    print(f"{len(sample)} moments from {len(runs)} runs {modes}, timeout {a.timeout:.0f} s\n")

    rows, summary = [], {}
    for model in a.models:
        if model not in installed and f"{model}:latest" not in installed:   # "phi4-mini" is "phi4-mini:latest"
            print(f"{model}: not pulled (ollama pull {model})")
            continue
        cfg = LLMReasonConfig(model=model)
        _, err, load_ms = ask(cfg, {"mode": "hover"}, timeout=300)
        lat, sc = [], []
        for i, p in enumerate(sample):
            reason, err, ms = ask(cfg, p, a.timeout)
            s = score(p, reason, err)
            lat.append(ms)
            sc.append(s)
            rows.append({"model": model, "i": i, "mode": p["mode"], "battery": p["battery"],
                         "ms": round(ms), "error": err, "reason": reason, **s})
        m = {k: round(float(np.mean([s[k] for s in sc])), 2) for k in sc[0]}
        ok = [l for l, s in zip(lat, sc) if not s["timeout"]]
        summary[model] = {"load_s": round(load_ms / 1000, 1),
                          "median_ms": round(float(np.median(ok))) if ok else None,
                          "p95_ms": round(float(np.percentile(ok, 95))) if ok else None, **m}
        print(f"{model}: load {summary[model]['load_s']} s, "
              f"median {summary[model]['median_ms']} ms, timeouts {m['timeout']:.0%}, "
              f"valid {m['valid']:.0%}, on-mode {m['on_mode']:.0%}, "
              f"false battery {m['false_bat']:.0%}, short {m['short']:.0%}")
        for r in [r for r in rows if r["model"] == model][:3]:
            print(f"    [{r['mode']}] {r['reason'] or r['error']}")

    with open(out.path("llm_bench.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["model"])
        w.writeheader()
        w.writerows(rows)
    out.record("summary", summary)
    out.write()
    print(f"\nwrote {out.dir}")


if __name__ == "__main__":
    main()
