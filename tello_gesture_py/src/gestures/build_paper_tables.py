"""Regenerate every table in the manuscript directly from the run logs.

Nothing in the paper is typed by hand: run this and paste, or \\input the
generated .tex fragments. Re-running after new flights updates every number
consistently, so a table can never drift away from the data behind it.

    python -m tello_gesture_py.src.gestures.build_paper_tables --outdir outputs/paper
"""

from tello_gesture_py.src.gestures.evidence import paper_glob
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "CENTER", 1: "LEFT", 2: "RIGHT", 3: "UP",
               4: "DOWN", 5: "FORWARD", 6: "BACK"}

SCENARIO_LABEL = {
    "authorized_gesture_accepted": "Authorized gesture accepted",
    "unauthorized_gesture_rejected": "Unauthorized gesture rejected",
    "face_following_activates": "Face-following activates",
    "gesture_preempts_face": "Gesture preempts face-following",
    "hysteresis_prevents_flicker": "Hysteresis prevents flicker",
    "search_starts_after_loss": "Search starts after target loss",
    "target_reacquired_after_search": "Target reacquired during search",
    "battery_failsafe_landing": "Battery failsafe landing",
}

IN_FLIGHT = ["authorized_gesture_accepted", "face_following_activates",
             "search_starts_after_loss", "target_reacquired_after_search",
             "battery_failsafe_landing"]
WEBCAM = ["unauthorized_gesture_rejected", "gesture_preempts_face",
          "hysteresis_prevents_flicker"]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def load_runs():
    """Every run directory with its manifest and logs."""
    out = []
    for d in sorted(paper_glob("outputs/runs/*/")):
        name = os.path.basename(os.path.normpath(d))
        try:
            man = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
        except Exception:
            man = {}

        def rd(fn):
            try:
                return pd.read_csv(os.path.join(d, fn))
            except Exception:
                return pd.DataFrame()

        out.append({
            "name": name, "dir": d, "manifest": man,
            "telemetry": rd("telemetry.csv"), "perf": rd("perf.csv"),
            "events": rd("perf_events.csv"), "scenarios": rd("scenarios.csv"),
            "decisions": rd("decisions.csv"),
        })
    return out


def arm_of(man):
    """Which identity-gate configuration a run used."""
    params = man.get("params") or {}
    cfg = man.get("config") or {}
    hy = params.get("auth_hysteresis", cfg.get("faceid_hysteresis"))
    return "hysteresis" if hy else "bare"


def collect_trials(runs):
    rows = []
    for r in runs:
        s = r["scenarios"]
        if not len(s):
            continue
        if "webcam_smoke" in r["name"]:
            continue
        man = r["manifest"]
        harness = man.get("harness", "controller")
        dec = r["decisions"]
        for _, t in s.iterrows():
            # Validity rule, applied uniformly: a trial only counts if the
            # aircraft was actually airborne inside the trial window. Catches
            # trials opened before takeoff; never selects on outcome.
            airborne = None
            if harness == "controller" and len(dec):
                w = dec[(dec["t"] >= t["t_start"]) & (dec["t"] <= t["t_end"])]
                airborne = bool(len(w) and w["command"].astype(str).str.startswith("rc ").any())
            rows.append({
                "run": r["name"], "harness": harness, "arm": arm_of(man),
                "scenario": t["scenario"], "success": bool(t["success"]),
                "duration_s": t.get("duration_s"), "airborne": airborne,
            })
    d = pd.DataFrame(rows)
    # A self-enrolled run cannot furnish an impostor-rejection trial.
    d = d[~((d["scenario"] == "unauthorized_gesture_rejected") & d["run"].str.contains("drone_home-flight-2"))]
    # Scenario 3 has a few trials from the pre-fix bench run: different config.
    d = d[~((d["scenario"] == "face_following_activates") & (d["arm"] == "bare"))]
    d["valid"] = d["airborne"].fillna(True)
    return d


def tex_escape(s):
    return str(s).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def table_scenarios_flight(trials):
    d = trials[(trials["harness"] == "controller") & trials["valid"]]
    lines = [r"\begin{table}[t]", r"\caption{In-flight scenario validation (deployed configuration).}",
             r"\label{tab:flight}", r"\centering", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Scenario & $n$ & Passed & 95\% CI \\", r"\midrule"]
    tot_n = tot_k = 0
    for sc in IN_FLIGHT:
        g = d[d["scenario"] == sc]
        n, k = len(g), int(g["success"].sum())
        if not n:
            continue
        tot_n += n
        tot_k += k
        lo, hi = wilson(k, n)
        lines.append(f"{SCENARIO_LABEL[sc]} & {n} & {k} & [{100*lo:.0f}, {100*hi:.0f}] \\\\")
    lines += [r"\midrule", f"Total & {tot_n} & {tot_k} & \\\\", r"\bottomrule",
              r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_scenarios_webcam(trials):
    d = trials[trials["harness"] == "webcam_demo"]
    lines = [r"\begin{table}[t]",
             r"\caption{Arbitration verification on the webcam harness, both identity-gate configurations.}",
             r"\label{tab:webcam}", r"\centering", r"\begin{tabular}{lcc}", r"\toprule",
             r"Scenario & Bare threshold & With hysteresis \\", r"\midrule"]
    for sc in WEBCAM:
        cells = []
        for arm in ("bare", "hysteresis"):
            g = d[(d["scenario"] == sc) & (d["arm"] == arm)]
            cells.append(f"{int(g['success'].sum())}/{len(g)}" if len(g) else "--")
        lines.append(f"{SCENARIO_LABEL[sc]} & {cells[0]} & {cells[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_runtime(runs):
    """Stage latency conditioned on the stage actually running, in flight."""
    # Choose the longest in-flight run in which the operator was actually
    # present: a run with nobody in frame never exercises face embedding, so
    # its latency table would have a hole where the heaviest stage belongs.
    best = None
    for r in runs:
        t, p = r["telemetry"], r["perf"]
        if not len(t) or not len(p) or "h" not in t:
            continue
        air = t[t["h"] > 0]
        if len(air) < 5:
            continue
        pf = p[(p["t"] >= air["t"].min()) & (p["t"] <= air["t"].max())]
        if "face_id_ms" not in pf.columns:
            continue
        n_faceid = int(pd.to_numeric(pf["face_id_ms"], errors="coerce").notna().sum())
        if n_faceid < 100:
            continue
        span = air["t"].max() - air["t"].min()
        if best is None or span > best[0]:
            best = (span, r)
    if best is None:
        return "% no in-flight run found", {}
    span, r = best
    t, p = r["telemetry"], r["perf"]
    air = t[t["h"] > 0]
    pf = p[(p["t"] >= air["t"].min()) & (p["t"] <= air["t"].max())]

    stages = [("hand_detect_ms", "Hand detection (MediaPipe Hands)"),
              ("face_detect_ms", "Face detection (MediaPipe)"),
              ("face_id_ms", "Face embedding + cosine match"),
              ("mode_manager_ms", "Mode-manager update"),
              ("command_exec_ms", "Command execution"),
              ("frame_age_ms", "Frame age (decode + queue)"),
              ("camera_to_command_ms", "End-to-end camera-to-command")]
    lines = [r"\begin{table}[t]",
             r"\caption{Per-stage latency in flight $\downarrow$, in ms, over the frames where the stage ran. Duty is that share.}",
             r"\label{tab:runtime}", r"\centering", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Stage & Mean & Median & p95 & Duty \\", r"& (ms) & (ms) & (ms) & (\%) \\", r"\midrule"]
    stats = {}
    for col, label in stages:
        if col not in pf.columns:
            continue
        x = pd.to_numeric(pf[col], errors="coerce").dropna()
        if x.empty:
            continue
        ran = pf.get(col + "_ran")
        duty = 100 * pd.to_numeric(ran, errors="coerce").fillna(0).mean() if ran is not None else 100.0
        stats[col] = (x.mean(), x.median(), np.percentile(x, 95), duty)
        lines.append(f"{label} & {x.mean():.2f} & {x.median():.2f} & {np.percentile(x,95):.2f} & {duty:.0f} \\\\")
    fps = (len(pf) - 1) / max(pf["t"].max() - pf["t"].min(), 1e-9)
    lines += [r"\midrule", f"Video stream & \\multicolumn{{4}}{{r}}{{{fps:.1f} fps}} \\\\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    stats["_fps"] = fps
    stats["_run"] = r["name"]
    stats["_airborne_s"] = span
    stats["_frames"] = len(pf)
    return "\n".join(lines), stats


def table_cross_session():
    pairs = [("AB", "A", "B"), ("AC", "A", "C"), ("BC", "B", "C")]
    data = {}
    for key, a, b in pairs:
        f = f"outputs/metrics/cross_session_{key}.json"
        if os.path.exists(f):
            data[key] = json.load(open(f, encoding="utf-8"))
    if not data:
        return "% no cross-session results", {}
    lines = [r"\begin{table}[t]",
             r"\caption{Cross-session gesture classification. Per-class $F_1$; the in-session figure is a random split within one recording session and is reported only as a reference point.}",
             r"\label{tab:crosssession}", r"\centering", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Class & A$\rightarrow$B & A$\rightarrow$C & B$\rightarrow$C \\", r"\midrule"]
    for i in range(7):
        vals = [data[k]["report"].get(str(i), {}).get("f1-score", float("nan")) for k, _, _ in pairs]
        lines.append(f"{CLASS_NAMES[i]} & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")
    lines.append(r"\midrule")
    lines.append("Macro $F_1$ & " + " & ".join(f"{data[k]['macro_f1']:.3f}" for k, _, _ in pairs) + r" \\")
    lines.append("Accuracy & " + " & ".join(f"{data[k]['accuracy']:.3f}" for k, _, _ in pairs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines), {k: data[k]["accuracy"] for k in data}


def table_ablation(runs):
    want = {"drone_n3-baseline": "Deployed configuration",
            "drone_n2-ablation-no-gate": "Identity gate disabled",
            "drone_n2-ablation-no-hysteresis": "Authorization hysteresis disabled"}
    found = {}
    for r in runs:
        for key, label in want.items():
            if key in r["name"] and len(r["decisions"]) > 10:
                found[label] = r
    if not found:
        return "% no ablation runs", {}
    lines = [r"\begin{table}[t]",
             r"\caption{In-flight ablations. Mode-switch rate under repeated operator entry and exit from the field of view.}",
             r"\label{tab:ablation}", r"\centering", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Configuration & Span (s) & Switches/min & Auth.\ flips \\", r"\midrule"]
    stats = {}
    for label in ["Deployed configuration", "Identity gate disabled", "Authorization hysteresis disabled"]:
        if label not in found:
            continue
        d = found[label]["decisions"].sort_values("t")
        span = d["t"].max() - d["t"].min()
        m = d["mode"].values
        sw = int((m[1:] != m[:-1]).sum())
        a = d["face_auth"].astype(bool).values
        fl = int((a[1:] != a[:-1]).sum())
        stats[label] = 60 * sw / span
        lines.append(f"{label} & {span:.0f} & {60*sw/span:.1f} & {fl} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines), stats


def key_facts(runs, trials):
    """Numbers quoted in the prose, so they also come from the logs."""
    f = {}
    f["n_runs"] = len(runs)
    f["n_trials"] = int(trials["valid"].sum())
    f["n_frames"] = int(sum(len(r["perf"]) for r in runs))

    # failsafe ladder
    for r in runs:
        if "drone_n4-failsafe-ladder" in r["name"]:
            e = r["events"]
            fs = e[e["event"] == "battery_failsafe_land"] if len(e) else []
            t = r["telemetry"]
            f["failsafe_fired"] = len(fs)
            if len(t):
                f["failsafe_bat_hi"] = float(t["bat"].iloc[0])
                f["failsafe_bat_lo"] = float(t["bat"].iloc[-1])
                air = t[t["h"] > 0]
                f["failsafe_airborne_s"] = float(air["t"].max() - air["t"].min()) if len(air) > 1 else 0

    # authorization rate by capture condition
    def auth(pat):
        hits = sorted(paper_glob(f"outputs/runs/*{pat}*/decisions.csv"))
        if not hits:
            return None
        d = pd.read_csv(hits[-1])
        d = d[(d["faceid_enrolled"] == True) & (d["faceid_score"] > -1)]  # noqa: E712
        return (100 * d["face_auth"].mean(), len(d)) if len(d) else None
    for label, pat in [("webcam", "webcam_smoke"), ("drone_ground", "drone_bench-2"), ("drone_flight", "drone_home-flight-1")]:
        v = auth(pat)
        if v:
            f[f"auth_{label}"] = v[0]
            f[f"auth_{label}_n"] = v[1]

    # search sweep coverage before and after retuning
    for tag, pat in [("sweep_old", "drone_home-flight-1"), ("sweep_new", "drone_n1-scenarios-6-7")]:
        for r in runs:
            if pat not in r["name"] or not len(r["telemetry"]) or not len(r["decisions"]):
                continue
            t, d = r["telemetry"], r["decisions"].sort_values("t")
            eps, cur = [], None
            for _, row in d.iterrows():
                if row["mode"] == "search_360":
                    cur = [row["t"], row["t"]] if cur is None else [cur[0], row["t"]]
                elif cur is not None:
                    eps.append(cur)
                    cur = None
            degs = []
            for a, b in eps:
                sub = t[(t["t"] >= a) & (t["t"] <= b)]
                if len(sub) < 3 or (b - a) < 2.0:
                    continue
                if not (sub["h"] > 0).any():   # RC is not sent on the ground
                    continue
                y = sub["yaw"].values.astype(float)
                degs.append(abs(float(np.sum((np.diff(y) + 180) % 360 - 180))))
            if degs:
                f[tag] = float(np.median(degs))
                f[tag + "_n"] = len(degs)
                break
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="outputs/paper")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = load_runs()
    trials = collect_trials(runs)

    runtime_tex, runtime_stats = table_runtime(runs)
    cross_tex, cross_stats = table_cross_session()
    abl_tex, abl_stats = table_ablation(runs)

    outputs = {
        "tab_flight.tex": table_scenarios_flight(trials),
        "tab_webcam.tex": table_scenarios_webcam(trials),
        "tab_runtime.tex": runtime_tex,
        "tab_crosssession.tex": cross_tex,
        "tab_ablation.tex": abl_tex,
    }
    for fn, body in outputs.items():
        with open(os.path.join(args.outdir, fn), "w", encoding="utf-8") as fh:
            fh.write(body + "\n")
        print(f"wrote {args.outdir}/{fn}")

    facts = key_facts(runs, trials)
    facts["runtime"] = {k: v for k, v in runtime_stats.items() if not k.startswith("_")}
    facts["runtime_meta"] = {k: runtime_stats[k] for k in runtime_stats if k.startswith("_")}
    facts["cross_session"] = cross_stats
    facts["ablation_switches_per_min"] = abl_stats
    with open(os.path.join(args.outdir, "key_facts.json"), "w", encoding="utf-8") as fh:
        json.dump(facts, fh, indent=2, default=str)
    print(f"wrote {args.outdir}/key_facts.json")
    print()
    print(json.dumps(facts, indent=2, default=str))


if __name__ == "__main__":
    main()
