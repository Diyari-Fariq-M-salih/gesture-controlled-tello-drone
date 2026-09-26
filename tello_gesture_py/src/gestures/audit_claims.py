"""Recompute every quantitative claim in the manuscript from the released logs.

The paper states that all quantitative claims are generated from the logs. This
script is what makes that statement checkable rather than asserted: each claim in
the prose is registered below with the value printed in the manuscript and a
function that rederives it from `outputs/`. Anything that cannot be rederived is
reported as UNVERIFIABLE rather than quietly passing.

    python -m tello_gesture_py.src.gestures.audit_claims
    python -m tello_gesture_py.src.gestures.audit_claims --out_json outputs/metrics/claim_audit.json

Exit status is non-zero if any claim fails or cannot be derived, so the audit can
gate a release.
"""

from tello_gesture_py.src.gestures.evidence import paper_glob
import argparse
import glob
import json
import math
import os
from typing import Optional

import numpy as np
import pandas as pd

DEPTH = {"FORWARD", "BACK"}

# Runs whose logs carry no telemetry are webcam-harness runs, not flights. Every
# aggregate that describes the aircraft has to exclude them or it is describing
# a laptop camera.
def _is_flight_run(rundir: str) -> bool:
    t = os.path.join(rundir, "telemetry.csv")
    if not os.path.exists(t):
        return False
    try:
        return len(pd.read_csv(t)) > 0
    except Exception:
        return False


def _campaign_runs():
    """Runs from the original campaign, before explicit classifier selection.

    Several claims describe that campaign specifically -- its run count, its
    radio conditions, and the three traces establishing which classifier it
    flew. The re-measurement flights are a separate study and must not be
    pooled into them.
    """
    out = []
    for d in _runs():
        try:
            m = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
        except Exception:
            continue
        if "classifier" not in m:
            out.append(d)
    return out


def _runs():
    """Run directories that actually produced data.

    RunContext creates its directory before the SDK handshake, so a launch that
    fails to reach the drone leaves an empty directory behind. Counting those as
    runs inflates every total derived from them.
    """
    out = []
    for d in sorted(paper_glob("outputs/runs/*/")):
        if os.path.exists(os.path.join(d, "manifest.json")):
            out.append(os.path.normpath(d))
    return out


def _all_run_dirs():
    return [os.path.normpath(d) for d in sorted(paper_glob("outputs/runs/*/"))]


def _read(rundir: str, fn: str) -> pd.DataFrame:
    p = os.path.join(rundir, fn)
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _exact(pat: str) -> Optional[str]:
    """The run directory whose tag is exactly `pat` (not a prefix of another)."""
    hits = [d for d in _runs() if os.path.basename(d).split("_", 1)[-1] == pat]
    return hits[-1] if hits else None


def _tagged(pat: str) -> Optional[str]:
    hits = [d for d in _runs() if pat in os.path.basename(d)]
    return hits[-1] if hits else None


def _scored(d: pd.DataFrame) -> pd.DataFrame:
    """Decision rows where the identity gate actually produced a score."""
    if d.empty or "faceid_enrolled" not in d:
        return pd.DataFrame()
    return d[(d["faceid_enrolled"] == True) & (d["faceid_score"] > -1)]  # noqa: E712


# --------------------------------------------------------------- derivations
def d_counts():
    """Trial counts under the same validity rule the paper tables use."""
    from .build_paper_tables import collect_trials, load_runs
    t = collect_trials(load_runs())
    flight = t[(t["harness"] == "controller") & t["valid"]]
    # Ground-capture runs carry no scenario trials, so they are counted apart
    # from the instrumented flight runs rather than pooled with them.
    GROUND = ("operational_gesture_eval", "stream_capture", "auto_collect_dataset")
    n_ground = 0
    for d in _runs():
        try:
            m = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
        except Exception:
            m = {}
        if m.get("harness") in GROUND:
            n_ground += 1
    n_controller = sum(
        1 for d in _runs()
        if (json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
            .get("harness", "controller")) == "controller")
    campaign = [d for d in _campaign_runs()
                if (json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
                    .get("harness") not in GROUND)]
    return {"n_runs": len(campaign), "n_ground_runs": n_ground,
            "n_flight_runs": n_controller,
            "n_empty_dirs": len(_all_run_dirs()) - len(_runs()),
            "n_trials": int(t["valid"].sum()),
            "n_flight_trials": int(len(flight)),
            "n_flight_passed": int(flight["success"].sum())}


def d_within_session():
    f = "outputs/metrics/per_session_eval.json"
    if not os.path.exists(f):
        return None
    j = json.load(open(f, encoding="utf-8"))
    return {k: round(j[k]["accuracy"], 3) for k in sorted(j)}


def d_loso():
    f = "outputs/metrics/loso_eval.json"
    if not os.path.exists(f):
        return None
    j = json.load(open(f, encoding="utf-8"))["loso"]
    accs = {k: round(j[k]["accuracy"], 3) for k in sorted(j)}
    accs["mean"] = round(float(np.mean([j[k]["accuracy"] for k in j])), 3)
    accs["test_n"] = {k: j[k]["test_n"] for k in sorted(j)}
    return accs


def d_transfer():
    f = "outputs/metrics/loso_eval.json"
    if not os.path.exists(f):
        return None
    t = json.load(open(f, encoding="utf-8"))["transfer"]
    return {f"{a}->{b}": round(v, 3) for a, row in t.items() for b, v in row.items()}


def d_sweep():
    out = {}
    for cap in ("100", "250", "400"):
        f = f"outputs/sampling_density/{cap}_per_class/training_metrics.json"
        if os.path.exists(f):
            out[cap] = round(json.load(open(f, encoding="utf-8"))["accuracy"], 4)
    return out or None


def d_capture_conditions():
    """Per-session photometry, sampling every 10th frame of every class.

    Reports grey level and the per-channel means. The blue-minus-red separation
    is the quantity the capture design turns on: session A was shot against a
    blue wall, which separates hand from background in hue, and sessions B and C
    were chosen to remove that cue.
    """
    import cv2
    dirs = {"A": "data/raw", "B": "data/raw/sessionB", "C": "data/raw/sessionC",
            "D": "data/raw/sessionD"}
    out = {}
    for key, root in dirs.items():
        cls = sorted(d for d in glob.glob(os.path.join(root, "0*")) if os.path.isdir(d))
        grey, ch = [], []
        for c in cls:
            for p in sorted(glob.glob(os.path.join(c, "*.jpg")))[::10]:
                im = cv2.imread(p)
                if im is None:
                    continue
                grey.append(float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean()))
                ch.append([float(im[:, :, i].mean()) for i in range(3)])
        if not grey:
            continue
        b, g, r = np.mean(ch, axis=0)
        out[key] = {"frames_sampled": len(grey), "grey": round(float(np.mean(grey))),
                    "blue": round(b), "green": round(g), "red": round(r),
                    "blue_minus_red": round(b - r)}
    # The images show the operator's face and are not published, so the four
    # sessions' statistics are kept beside the metrics. Where the images are
    # present they are recomputed and the record refreshed; where they are not
    # (any public clone) the record is what gets checked.
    rec = "outputs/metrics/capture_conditions.json"
    if len(out) == len(dirs):
        with open(rec, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
    elif os.path.exists(rec):
        with open(rec, encoding="utf-8") as f:
            out = json.load(f)
    return out or None


def d_auth_rates():
    """False rejection through each capture path, at the deployed threshold."""
    out = {}
    for label, pat in (("webcam", "webcam_smoke"), ("drone_ground", "drone_bench-2"),
                       ("drone_flight", "drone_home-flight-1")):
        r = _exact(pat) or _tagged(pat)
        if not r:
            continue
        d = _scored(_read(r, "decisions.csv"))
        if len(d):
            out[label] = {"n": len(d), "frr_pct": round(100 * (1 - d["face_auth"].mean()), 1)}
    return out or None


def d_stale_crop():
    """The stale-crop correction, before (bench) and after (bench2)."""
    out = {}
    for key, tag in (("bench", "drone_bench"), ("bench2", "drone_bench-2")):
        r = _exact(tag)
        if not r:
            continue
        d = _read(r, "decisions.csv")
        s = _scored(d)["faceid_score"]
        if not len(s):
            continue
        gen, bg = s[s >= 0.4], s[s < 0.4]
        out[key] = {
            "n": len(s),
            "below_0.4_pct": round(100 * len(bg) / len(s)),
            "authorized_pct": round(100 * (s >= 0.55).mean() * 100) / 100,
            "genuine_mean": round(float(gen.mean()), 3),
            "genuine_sd": round(float(gen.std()), 3),
            "background_mean": round(float(bg.mean()), 3),
            "background_sd": round(float(bg.std()), 3),
            "modes": d["mode"].value_counts().to_dict(),
        }
    return out or None


def d_impostor():
    """Impostor authorization inside impostor trials, both gate configurations."""
    out = {}
    for tag, arm in (("webcam_d1-unauthorised", "bare"), ("webcam_d1-unauthorised-hysteresis", "hysteresis")):
        r = _exact(tag)
        if not r:
            continue
        d, s = _read(r, "decisions.csv"), _read(r, "scenarios.csv")
        if d.empty or s.empty:
            continue
        s = s[s["scenario"] == "unauthorized_gesture_rejected"]
        tot = auth = 0
        for _, t in s.iterrows():
            w = _scored(d[(d["t"] >= t["t_start"]) & (d["t"] <= t["t_end"])])
            tot += len(w)
            auth += int(w["face_auth"].sum()) if len(w) else 0
        out[arm] = {"trials": len(s), "passed": int(s["success"].sum()),
                    "frames": tot, "authorized_pct": round(100 * auth / max(tot, 1), 1)}
    return out or None


def d_hysteresis_trial():
    """The single failed preemption trial, and the Schmitt trigger replayed on it.

    The replay applies Eq. (1) to the recorded score sequence with the deployed
    constants, so the counterfactual is computed from the log rather than
    asserted. It changes no state in the running system.
    """
    ON, OFF, K = 0.55, 0.45, 3
    for p in sorted(paper_glob("outputs/runs/*/scenarios.csv")):
        r = os.path.dirname(p)
        sdf = pd.read_csv(p)
        f = sdf[(sdf["scenario"] == "gesture_preempts_face") & (~sdf["success"].astype(bool))]
        if not len(f):
            continue
        d = _read(r, "decisions.csv")
        t = f.iloc[0]
        w = d[(d["t"] >= t["t_start"]) & (d["t"] <= t["t_end"])].copy()
        if w.empty:
            continue
        w["rel"] = w["t"] - t["t_start"]
        scores = [round(float(x), 3) for x in w["faceid_score"]]

        hand = w[w["hand_raw"].astype(bool)]
        gest = w[w["mode"] == "gesture"]
        # Median preemption latency over the successful trials *in the same
        # arm*. Comparing against the hysteresis arm would compare against the
        # fix, not against the baseline the failure belongs to.
        good = []
        for _, tt in sdf[(sdf["scenario"] == "gesture_preempts_face")
                         & (sdf["success"].astype(bool))].iterrows():
            ww = d[(d["t"] >= tt["t_start"]) & (d["t"] <= tt["t_end"])]
            g = ww[ww["mode"] == "gesture"]
            if len(g):
                good.append(float(g["t"].iloc[0] - tt["t_start"]))

        auth, streak, replay = False, 0, []
        for x in scores:
            if x >= ON:
                auth, streak = True, 0
            elif x < OFF:
                streak += 1
                if streak >= K:
                    auth = False
            else:
                streak = 0
            replay.append(auth)
        first = next((w["rel"].iloc[i] for i, a in enumerate(replay)
                      if a and bool(w["hand_raw"].iloc[i])), None)
        return {
            "run": os.path.basename(r), "frames": len(w), "scores": scores,
            "hand_raised_at_s": round(float(hand["rel"].iloc[0]), 2) if len(hand) else None,
            "preempted_at_s": round(float(gest["rel"].iloc[0]), 2) if len(gest) else None,
            "median_preemption_s_successful_trials": round(float(np.median(good)), 2) if good else None,
            "n_successful_same_arm": len(good),
            "authorized_as_logged": int(w["face_auth"].astype(bool).sum()),
            "authorized_under_replay": int(sum(replay)),
            "replay_preemption_s": round(float(first), 2) if first is not None else None,
        }
    return None


def d_ablation():
    """Mode-switch rate per minute under each identity-gate configuration."""
    want = {"drone_n3-baseline": "deployed", "drone_n2-ablation-no-gate": "no_gate",
            "drone_n2-ablation-no-hysteresis": "no_auth_hysteresis"}
    out = {}
    for r in _runs():
        for key, label in want.items():
            if key in os.path.basename(r):
                d = _read(r, "decisions.csv").sort_values("t")
                if len(d) < 10:
                    continue
                span = d["t"].max() - d["t"].min()
                m = d["mode"].values
                out[label] = round(60 * int((m[1:] != m[:-1]).sum()) / span, 1)
    return out or None


def d_search():
    """Yaw actually covered per search episode, before and after retuning."""
    out = {}
    for label, pat in (("before", "drone_home-flight-1"), ("after", "drone_n1-scenarios-6-7")):
        degs, durs, runs = [], [], []
        # Pool every run in the configuration: one run is not the unit of
        # interest, a search episode is, and picking a single run by hand is
        # the kind of choice this audit exists to remove.
        for r in [x for x in _runs() if pat in os.path.basename(x)]:
            t, d = _read(r, "telemetry.csv"), _read(r, "decisions.csv")
            if t.empty or d.empty:
                continue
            d = d.sort_values("t")
            eps, cur = [], None
            for _, row in d.iterrows():
                if row["mode"] == "search_360":
                    cur = [row["t"], row["t"]] if cur is None else [cur[0], row["t"]]
                elif cur is not None:
                    eps.append(cur)
                    cur = None
            if cur:
                eps.append(cur)
            n0 = len(degs)
            for a, b in eps:
                sub = t[(t["t"] >= a) & (t["t"] <= b)]
                if len(sub) < 3 or (b - a) < 2.0 or not (sub["h"] > 0).any():
                    continue
                y = sub["yaw"].values.astype(float)
                degs.append(abs(float(np.sum((np.diff(y) + 180) % 360 - 180))))
                durs.append(b - a)
            if len(degs) > n0:
                runs.append(os.path.basename(r))
        if degs:
            out[label] = {"episodes": len(degs), "runs": runs,
                          "median_deg": round(float(np.median(degs))),
                          "median_duration_s": round(float(np.median(durs)), 1)}
    return out or None


def d_failsafe():
    r = _tagged("drone_n4-failsafe-ladder")
    if not r:
        return None
    t, e = _read(r, "telemetry.csv"), _read(r, "perf_events.csv")
    if t.empty or e.empty:
        return None
    fs = e[e["event"] == "battery_failsafe_land"]
    bats = []
    for _, x in fs.iterrows():
        w = t[t["t"] <= x["t"]]
        if len(w):
            bats.append(float(w["bat"].iloc[-1]))
    air = t[t["h"] > 0]
    return {"landings": len(fs), "battery_first": bats[0] if bats else None,
            "battery_last": bats[-1] if bats else None,
            "airborne_s": round(float(air["t"].max() - air["t"].min()))}


def d_radio():
    """Stream rate and reconnections per flight run, and the battery correlation."""
    rows = []
    for r in _campaign_runs():
        if not _is_flight_run(r):
            continue
        p = _read(r, "perf.csv")
        if len(p) < 50:
            continue
        span = p["t"].max() - p["t"].min()
        e = _read(r, "perf_events.csv")
        reopen = int((e["event"] == "video_reopen").sum()) if len(e) and "event" in e else 0
        t = _read(r, "telemetry.csv")
        rows.append({"run": os.path.basename(r), "fps": (len(p) - 1) / max(span, 1e-9),
                     "reopen": reopen, "bat": float(t["bat"].mean()) if len(t) and "bat" in t else np.nan})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    m = df.dropna(subset=["bat"])
    return {"runs": len(df), "fps_min": round(float(df["fps"].min()), 1),
            "fps_max": round(float(df["fps"].max()), 1),
            "pearson_r_fps_battery": round(float(np.corrcoef(m["fps"], m["bat"])[0, 1]), 3),
            "worst": df.sort_values("fps").iloc[0].to_dict(),
            "most_reopens": df.sort_values("reopen").iloc[-1].to_dict()}


def d_depth_contamination():
    """Frames carrying an uncommanded depth component.

    Counted over authorized-gesture trials in which the operator never issued a
    depth gesture as the primary label. A frame is contaminated when its label
    carries FORWARD or BACK as a *secondary* component, e.g. DOWN-FORWARD: the
    commanded axis is correct and a depth velocity rides along with it.
    """
    act = cont = trials = 0
    for r in _runs():
        d, s = _read(r, "decisions.csv"), _read(r, "scenarios.csv")
        if d.empty or s.empty:
            continue
        for _, t in s[s["scenario"] == "authorized_gesture_accepted"].iterrows():
            w = d[(d["t"] >= t["t_start"]) & (d["t"] <= t["t_end"]) & (d["mode"] == "gesture")]
            g = w["gesture"].astype(str)
            g = g[g != "NOHAND"]
            if not len(g):
                continue
            if {x.split("-")[0] for x in g} & DEPTH:   # depth was deliberate here
                continue
            trials += 1
            act += len(g)
            cont += int(g.apply(lambda x: bool(set(x.split("-")[1:]) & DEPTH)).sum())
    return {"trials": trials, "active_frames": act, "contaminated": cont,
            "pct": round(100 * cont / max(act, 1), 1)}


def d_llm():
    """Explanation quality and latency, with both denominators kept separate.

    A reason string persists across decision rows between model calls, so the
    number of decision records carrying a reason is not the number of
    explanations produced. Conflating them inflates the sample by 2.5x.
    """
    r = _exact("drone_bench")
    if not r:
        return None
    d, e = _read(r, "decisions.csv"), _read(r, "perf_events.csv")
    if d.empty:
        return None
    s = d["llm_reason"].dropna().astype(str)
    s = s[s.str.strip().ne("")]
    lat = e[e["event"] == "llm_explanation"]["latency_ms"].dropna() if len(e) else pd.Series(dtype=float)
    # A spurious low-battery warning: the model emits the failsafe sentence while
    # the logged charge is far above the threshold it names.
    warn = d[d["llm_reason"].astype(str).str.contains("Battery warning: battery <= 15", na=False)]
    bats = sorted(warn["battery"].dropna().unique().tolist())
    hi = None
    for rr in _runs():
        dd = _read(rr, "decisions.csv")
        if dd.empty or "llm_reason" not in dd:
            continue
        w = dd[dd["llm_reason"].astype(str).str.contains("Battery warning: battery <= 15", na=False)]
        if len(w) and w["battery"].notna().any():
            v = float(w["battery"].max())
            hi = v if hi is None else max(hi, v)
    return {"decision_rows_with_reason": len(s), "explanations_logged": int(len(lat)),
            "hard_error_rows": int(s.str.contains("LLM error", na=False).sum()),
            "hard_error_pct_of_rows": round(100 * s.str.contains("LLM error", na=False).mean()),
            "latency_mean_ms": round(float(lat.mean())) if len(lat) else None,
            "latency_p95_ms": round(float(np.percentile(lat, 95))) if len(lat) else None,
            "spurious_battery_warning_charges_this_run": bats,
            "spurious_battery_warning_max_charge_any_run": hi}


def d_runtime():
    """Conditioned per-stage latency from the longest in-flight run."""
    best = None
    for r in _runs():
        t, p = _read(r, "telemetry.csv"), _read(r, "perf.csv")
        if t.empty or p.empty or "h" not in t or "face_id_ms" not in p.columns:
            continue
        air = t[t["h"] > 0]
        if len(air) < 5:
            continue
        pf = p[(p["t"] >= air["t"].min()) & (p["t"] <= air["t"].max())]
        if int(pd.to_numeric(pf["face_id_ms"], errors="coerce").notna().sum()) < 100:
            continue
        span = air["t"].max() - air["t"].min()
        if best is None or span > best[0]:
            best = (span, r, pf)
    if best is None:
        return None
    span, r, pf = best
    out = {"run": os.path.basename(r), "airborne_s": round(span), "frames": len(pf),
           "fps": round((len(pf) - 1) / max(pf["t"].max() - pf["t"].min(), 1e-9), 1)}
    for c in ("camera_to_command_ms", "frame_age_ms", "hand_detect_ms", "face_id_ms"):
        x = pd.to_numeric(pf[c], errors="coerce").dropna()
        if len(x):
            out[c] = {"median": round(float(x.median()), 1), "p95": round(float(np.percentile(x, 95)), 1)}
    return out


def d_envelope():
    """Authorization against apparent face size: the bin counts behind Figure 2."""
    r = _tagged("drone_distance")
    if not r:
        return None
    d = _read(r, "decisions.csv")
    if d.empty or "face_bbox_h" not in d:
        return None
    d = d[(d["faceid_enrolled"] == True) & (d["face_raw"] == True) & d["face_bbox_h"].notna()]  # noqa: E712
    bins = [0, 150, 180, 220, 280, 350, 480, 620]
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = d[(d["face_bbox_h"] >= lo) & (d["face_bbox_h"] < hi)]
        if len(g):
            out[f"{lo}-{hi}"] = {"n": len(g), "auth_pct": round(100 * g["face_auth"].mean())}
    return {"run": os.path.basename(r), "total_frames": len(d), "bins": out}


def d_rule_operational():
    """The deployed rule path on the vocabulary it decodes."""
    f = "outputs/metrics/operational_rule.json"
    if not os.path.exists(f):
        return None
    j = json.load(open(f, encoding="utf-8"))
    d = [x for x in j["classifiers"] if "deployed" in x["classifier"]][0]
    u = [x for x in j["classifiers"] if "ungated" in x["classifier"]]
    ev = j.get("depth_events", {})
    return {"accuracy": round(d["accuracy_all_frames"], 3),
            "hold_accuracy": round(d["hold_accuracy"], 3),
            "frames": d["frames"], "holds": d["holds"],
            "f1": {k: round(v["f1"], 3) for k, v in d["per_class"].items()},
            "gate_delta": round(abs(d["accuracy_all_frames"] - u[0]["accuracy_all_frames"]), 3) if u else None,
            "depth_hit_rate": ev.get("FORWARD", {}).get("hold_hit_rate"),
            "depth_frame_share": ev.get("FORWARD", {}).get("frame_share"),
            "back_opposite_rate": ev.get("BACK", {}).get("opposite_rate"),
            "excluded_holds": sum(v["holds"] for v in j.get("declared_exclusions", {}).values()),
            # The exclusion rests on the inversion being total. If a later
            # capture made it partial the rule would no longer apply, so the
            # frame counts are registered rather than just the hold count.
            "excluded_frames_left": j.get("declared_exclusions", {})
                .get("LEFT:0-2", {}).get("frames"),
            "excluded_frames_right": j.get("declared_exclusions", {})
                .get("RIGHT:0-2", {}).get("frames")}


def d_inflight():
    """Hover-locked in-flight comparison, both arms."""
    import math as _m
    LAT = {"LEFT", "RIGHT", "UP", "DOWN"}

    def prim(x):
        if not isinstance(x, str):
            return None
        for p in x.split("-"):
            if p in LAT:
                return p
        return x.split("-")[0]

    def load(tag, rule):
        hits = [d for d in _runs() if tag in os.path.basename(d)]
        if not hits:
            return None
        d = _read(hits[-1], "cued.csv")
        if d.empty:
            return None
        h = d[d["phase"] == "hold"].copy()
        h["pred"] = h["emitted_gesture"].apply(prim) if rule else h["emitted_gesture"]
        return h

    svm = load("drone_n6-hover-cued-svm", False)
    rA = load("drone_n6-hover-cued-rule", True)
    rL = load("drone_n6-hover-lateral-rule", True)
    if svm is None or rA is None or rL is None:
        return None
    rule = pd.concat([rA[~rA["cue_name"].isin(["LEFT", "RIGHT"])], rL],
                     ignore_index=True)
    out = {}
    for nm, h in (("svm", svm), ("rule", rule)):
        ok = h["pred"] == h["cue_name"]
        holds = hit = 0
        for _, g in h.groupby(["round", "cue_name"]):
            holds += 1
            v = g[g["hand_detected"] == 1]["pred"].mode()
            if len(v) and v.iloc[0] == g["cue_name"].iloc[0]:
                hit += 1
        per_class, prec, f1, hold_pc = {}, {}, {}, {}
        for c in sorted(set(h["cue_name"])):
            cu = h[h["cue_name"] == c]
            tp = float((cu["pred"] == c).sum())
            rec = tp / len(cu)
            npred = float((h["pred"] == c).sum())
            p = tp / npred if npred else 0.0
            per_class[c] = round(rec, 3)
            prec[c] = round(p, 3)
            f1[c] = round(2 * rec * p / (rec + p), 3) if (rec + p) else 0.0
        # Per-hold recall by class. The paper argues the hold is the independent
        # unit, so a class where per-hold and per-frame diverge is exactly where
        # that argument does work -- and a description of one must not be
        # written as though it were the other.
        maj = {}
        for (rd, c), g in h.groupby(["round", "cue_name"]):
            v = g[g["hand_detected"] == 1]["pred"].mode()
            maj.setdefault(c, []).append(v.iloc[0] if len(v) else None)
        for c, ms in maj.items():
            hold_pc[c] = round(sum(m == c for m in ms) / len(ms), 3)
        out[nm] = {"frames": len(h), "accuracy": round(float(ok.mean()), 3),
                   "holds": holds, "hold_accuracy": round(hit / holds, 3),
                   "detection": round(float(h["hand_detected"].mean()), 3),
                   "per_class": per_class, "precision": prec, "f1": f1,
                   "per_class_hold": hold_pc,
                   "hold_majorities": {c: sorted(x for x in ms if x)
                                       for c, ms in maj.items()}}
        if nm == "svm":
            b = h[h["cue_name"] == "BACK"]
            out[nm]["back_frame_split"] = {
                k: round(float(v) / len(b), 3)
                for k, v in b["pred"].value_counts().items()}
    return out


def d_generalisations():
    """Prose sentences that quantify over a whole table.

    These are the claims the per-figure audit misses: each number in
    "four of five static classes exceed 0.92" is individually absent from the
    text, so nothing was checked, and the sentence was wrong for both columns
    for a full revision. Registered here as counts recomputed from the logs.
    """
    inf = d_inflight()
    if not inf:
        return None
    STATIC = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]
    out = {
        "svm_static_ge_092": sum(inf["svm"]["per_class"][c] >= 0.92 for c in STATIC),
        "rule_static_ge_092": sum(inf["rule"]["per_class"][c] >= 0.92 for c in STATIC),
        # The same sentence on the metric the table's caption argues for. These
        # two counts disagreeing is not an error; a sentence that does not say
        # which one it means is.
        "svm_static_f1_ge_092": sum(inf["svm"]["f1"][c] >= 0.92 for c in STATIC),
        "rule_static_below_092": sum(inf["rule"]["per_class"][c] < 0.92 for c in STATIC),
        "svm_static_min_recall": min(inf["svm"]["per_class"][c] for c in STATIC),
    }
    # "A hand was detected in every frame" -- ground SVM capture.
    f0 = "outputs/metrics/operational_gesture.json"
    if os.path.exists(f0):
        j0 = json.load(open(f0, encoding="utf-8"))
        sv0 = [x for x in j0["classifiers"] if x["classifier"] == "RBF-SVM"][0]
        out["ground_svm_detection_rate"] = sv0["detection_rate"]
    # ground SVM: cells above 0.10 off-diagonal
    f = "outputs/metrics/operational_gesture.json"
    if os.path.exists(f):
        j = json.load(open(f, encoding="utf-8"))
        svm = [x for x in j["classifiers"] if x["classifier"] == "RBF-SVM"][0]
        cm = np.array(svm["confusion_matrix_with_nodetect"], float)
        nrm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        out["ground_svm_offdiag_gt_010"] = int(sum(
            1 for i in range(7) for jj in range(8) if i != jj and nrm[i, jj] > 0.10))
    # held-out folds C and D: largest off-diagonal
    g = "outputs/metrics/loso_eval.json"
    if os.path.exists(g):
        L = json.load(open(g, encoding="utf-8"))["loso"]
        mx = 0.0
        for k in ("C", "D"):
            cm = np.array(L[k]["confusion_matrix"], float)
            nrm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            mx = max(mx, max(nrm[i, jj] for i in range(7) for jj in range(7) if i != jj))
        out["folds_CD_max_offdiag"] = round(float(mx), 3)
    # Gap decomposition. "82% of the gap is on the depth channel" is a claim
    # about a weighted sum of per-class recalls, and the weights are the frame
    # counts, which differ slightly between the two captures. Both weightings
    # are registered: the headline survives either, the split of the small
    # remainder between UP and CENTER does not, which is why the paper does not
    # apportion it.
    sv, ru = inf["svm"], inf["rule"]
    out["gap"] = round(sv["accuracy"] - ru["accuracy"], 3)
    d2 = d_inflight_frame_counts()
    if d2:
        ns, nr = d2["svm"], d2["rule"]
        ts, tr = sum(ns.values()), sum(nr.values())
        own = {c: ns[c] / ts * sv["per_class"][c] - nr[c] / tr * ru["per_class"][c]
               for c in ns}
        pooled = {c: (ns[c] + nr[c]) / (ts + tr)
                  * (sv["per_class"][c] - ru["per_class"][c]) for c in ns}
        for nm, dd in (("own", own), ("pooled", pooled)):
            tot = sum(dd.values())
            dep = dd["FORWARD"] + dd["BACK"]
            out["depth_share_" + nm] = round(dep / tot, 3)
            out["remainder_" + nm] = round(tot - dep, 3)
        out["gap_decomposition_own"] = {c: round(v, 4) for c, v in own.items()}
    return out


def d_inflight_frame_counts():
    """Cued frames per class in each arm, the weights the decomposition uses."""
    import csv as _csv
    LATC = {"LEFT", "RIGHT", "UP", "DOWN"}

    def cnt(tag, drop_lr=False):
        hits = [d for d in _runs() if tag in os.path.basename(d)]
        if not hits:
            return None
        d = _read(hits[-1], "cued.csv")
        if d.empty:
            return None
        h = d[d["phase"] == "hold"]
        if drop_lr:
            h = h[~h["cue_name"].isin(["LEFT", "RIGHT"])]
        return h["cue_name"].value_counts().to_dict()

    sv = cnt("drone_n6-hover-cued-svm")
    ra = cnt("drone_n6-hover-cued-rule", drop_lr=True)
    rl = cnt("drone_n6-hover-lateral-rule")
    if not sv or not ra or not rl:
        return None
    ru = dict(ra)
    for k, v in rl.items():
        ru[k] = ru.get(k, 0) + v
    return {"svm": sv, "rule": ru}


def d_ground_envelope_frr():
    """On-ground false rejection restricted to the working envelope.

    The unconditioned on-ground average the paper first reported was worse than
    the in-flight rate, which inverts the expected ordering. It was an artefact
    of a sample that included frames past the envelope, where the detector
    returns background rather than a face. Restricted to the marks inside the
    envelope the ordering is the expected one, and both figures are registered:
    the per-frame threshold comparison and the rate the deployed Schmitt gate
    actually produces on the same frames.
    """
    import glob as _glob
    frames = []
    for p in paper_glob("outputs/runs/*capture_envelope/stream_identity.csv"):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d = d[(d["face_raw"].astype(int) == 1) & d["faceid_score"].notna()]
    from ..config import ControllerConfig
    thr = ControllerConfig().faceid_cosine_thr
    e = d[d["distance_m"] <= 1.5]
    if len(e) < 100:
        return None
    return {"n": int(len(e)),
            "frr_pct_per_frame": round(100 * (1 - (e["faceid_score"] >= thr).mean()), 1),
            "frr_pct_gated": round(100 * (1 - e["face_auth"].astype(bool).mean()), 1),
            "threshold": thr}


def d_follow_standoff():
    """The face-following target framing, in metres, from the envelope sweep.

    `FaceFollowConfig.target_area_frac` is a fraction of frame area, which is
    not a distance until something measures the mapping. The sweep logs the face
    box and the operator's declared standoff at eight marks, so the calibration
    is measured through the deployed optics rather than estimated from a focal
    length and an assumed face size -- which is how this figure was first got
    wrong, by 0.8 m and in the direction that reversed its conclusion.

    Fitted only inside the envelope: past 1.75 m the detector returns background
    and the box stops being a face, so those marks would corrupt the fit.
    """
    import glob as _glob
    frames = []
    for p in paper_glob("outputs/runs/*capture_envelope/stream_identity.csv"):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    d = d[(d["face_raw"].astype(int) == 1) & d["face_bbox_h"].notna()
          & d["face_bbox_w"].notna() & d["distance_m"].notna()]
    fit = d[d["distance_m"] <= 1.75]
    if len(fit) < 100:
        return None

    # Read the two constants from the deployed dataclass, not from the paper.
    from ..face_follow import FaceFollowConfig
    fc = FaceFollowConfig()
    target = float(fc.target_area_frac)
    dead = float(fc.deadband_area)

    W, H = 960, 720
    area = fit["face_bbox_w"] * fit["face_bbox_h"] / float(W * H)
    k_area = float(np.median(np.sqrt(area) * fit["distance_m"]))
    k_h = float(np.median(fit["face_bbox_h"] * fit["distance_m"]))

    def dist(a):
        return k_area / math.sqrt(a)

    return {"target_area_frac": target,
            "deadband_area": dead,
            "n_frames_in_envelope": int(len(fit)),
            "k_sqrt_area_m": round(k_area, 4),
            "k_bbox_h_px_m": round(k_h, 1),
            "standoff_m": round(dist(target), 2),
            "band_near_m": round(dist(target + dead), 2),
            "band_far_m": round(dist(target - dead), 2),
            # The conclusion, not just the number: the target must sit inside
            # the envelope the gate admits, which is what the correction turned
            # on.
            "inside_envelope": bool(0.5 <= dist(target) <= 1.5)}


def d_design_table():
    """Check the rendered Table~III against the deployed configuration.

    `build_design_table` writes the table from the dataclasses, so this cannot
    fail while both are regenerated together -- which is the point: it fails when
    someone hand-edits the .tex, which is exactly how a specification drifts
    from the artefact it claims to specify.
    """
    import re as _re
    t = "paper/tables/tab_design.tex"
    if not os.path.exists(t):
        return None
    txt = open(t, encoding="utf-8").read()
    rows = {}
    for line in txt.splitlines():
        if "&" not in line or line.lstrip().startswith("\\"):
            continue
        cells = [c.strip() for c in line.rstrip("\\\\").split("&")]
        if len(cells) == 5 and "Parameter" not in cells[0]:
            rows[cells[0]] = cells[1]
            rows.setdefault("_results", []).append(cells[4])

    from ..config import ControllerConfig
    from ..face_follow import FaceFollowConfig
    c, f = ControllerConfig(), FaceFollowConfig()

    # (substring identifying the row, the value that must appear in it)
    want = [
        ("throttle", f"{c.hand_every_n}, {c.face_every_n}"),
        ("EMA", f"{c.ema_alpha}"),
        (r"theta_{\mathrm{dir}}", f"{c.dir_thr}"),
        (r"theta_{\mathrm{scale}}", f"{c.scale_thr}"),
        ("stability streak", f"{c.gesture_fb_streak_on}"),
        ("Face-follow target", f"{f.target_area_frac}"),
        (r"tau_{\mathrm{on}}", f"{c.faceid_cosine_thr}"),
        (r"tau_{\mathrm{off}}", f"{c.faceid_release_thr}, {c.faceid_release_frames}"),
        ("Enrolment", f"{c.faceid_enroll_samples}"),
        ("Delta_{\\max}", f"{f.crop_max_age_s}"),
        ("Mode hold", f"{c.mode_hold_s}"),
        ("Search sweep", f"{c.search_duration_s:.0f}"),
        ("Battery failsafe", f"{c.battery_land_pct}"),
        ("LLM decision rate", f"{c.llm_decision_hz:.0f}"),
    ]
    agree = 0
    bad = []
    for key, val in want:
        hit = [v for k, v in rows.items() if key in k]
        if hit and val in hit[0]:
            agree += 1
        else:
            bad.append(key)
    res = rows.pop("_results", [])
    return {"rows": len(rows), "params_matching_code": agree,
            "params_checked": len(want), "mismatched": bad,
            # The one row whose value is a calibration rather than a constant.
            "standoff_row_present": any("0.94" in r for r in res)}


def d_cross_section():
    """Claims that must agree across sections.

    Each entry is a quantity stated in one place and restated, in a different
    unit or a different decomposition, in another. The audit cannot see that two
    sentences are about the same thing, so the relationship is registered here
    rather than the numbers individually.
    """
    inf = d_inflight()
    if not inf:
        return None
    sv = inf["svm"]
    out = {
        # V-J describes the BACK holds; Table VII gives the per-frame recall.
        # A hold description implying a recall the table contradicts is the
        # error this check exists for.
        "back_hold_recall": sv["per_class_hold"]["BACK"],
        "back_frame_recall": sv["per_class"]["BACK"],
        "back_hold_minus_frame": round(sv["per_class_hold"]["BACK"]
                                       - sv["per_class"]["BACK"], 3),
        "back_majority_counts": {
            k: sv["hold_majorities"]["BACK"].count(k)
            for k in sorted(set(sv["hold_majorities"]["BACK"]))},
        # V-E states the split of BACK between LEFT and RIGHT; it must sum with
        # the recall to the whole class.
        "back_split": sv.get("back_frame_split", {}),
        "back_split_sums_to_one": round(sum(sv.get("back_frame_split", {}).values()), 3),
    }
    # Table I's headline cell is Table VII's per-frame SVM accuracy.
    t = "paper/tables/tab_related.tex"
    if os.path.exists(t):
        import re as _re
        txt = open(t, encoding="utf-8").read()
        m = _re.search(r"(0\.\d{3})[^&]*19\.3", txt)
        if m:
            out["table_I_cell"] = float(m.group(1))
    out["table_VII_frame_accuracy"] = sv["accuracy"]
    return out


def d_pair_factors():
    f = "outputs/metrics/loso_eval.json"
    if not os.path.exists(f):
        return None
    g = json.load(open(f, encoding="utf-8"))["pair_summary"]["groups"]
    return {k: g[k]["mean_accuracy"] for k in g}


def d_protocol_ladder():
    """The three protocols the abstract and conclusion should quote."""
    a = d_within_session()
    b = d_loso()
    f = "outputs/metrics/operational_gesture.json"
    op = None
    if os.path.exists(f):
        j = json.load(open(f, encoding="utf-8"))
        svm = [x for x in j["classifiers"] if x["classifier"] == "RBF-SVM"]
        if svm:
            op = round(svm[0]["accuracy_all_frames"], 3)
    if not a or not b:
        return None
    return {"within_A": a["A"], "loso_mean": b["mean"], "operational": op,
            "gap_within_minus_loso": round(a["A"] - b["mean"], 3),
            "gap_within_minus_operational": round(a["A"] - op, 3) if op else None}


def d_classifier_deployed():
    """Which gesture classifier the logged flights actually ran.

    Read three ways that cannot disagree with execution: the manifest field is a
    readout of the branch variable, the per-stage latency column is written
    inside the branch, and compound labels can only come from the rule path.
    """
    import csv as _csv
    field, svm_col, rule_col, compound = set(), 0, 0, 0
    for d in _campaign_runs():
        m = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
        if m.get("harness") in ("operational_gesture_eval", "stream_capture",
                                "auto_collect_dataset"):
            continue
        c = (m.get("model") or {}).get("classifier")
        if c:
            field.add(c)
        p = os.path.join(d, "perf.csv")
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    hdr = next(_csv.reader(fh))
                svm_col += "svm_infer_ms" in hdr
                rule_col += "rule_infer_ms" in hdr
            except Exception:
                pass
        dec = _read(d, "decisions.csv")
        if not dec.empty and "gesture" in dec:
            g = dec["gesture"].dropna().astype(str)
            compound += int(g.str.contains("-", na=False).sum())
    return {"manifest_classifier_values": sorted(field),
            "runs_with_svm_latency_column": svm_col,
            "runs_with_rule_latency_column": rule_col,
            "compound_labels_in_flight": compound}


def d_llm_denominators():
    """LLM figures with every denominator named."""
    r = _exact("drone_bench")
    if not r:
        return None
    d, e = _read(r, "decisions.csv"), _read(r, "perf_events.csv")
    s_ = d["llm_reason"].dropna().astype(str)
    s_ = s_[s_.str.strip().ne("")]
    calls = s_[s_ != s_.shift()]
    lat = e[e["event"] == "llm_explanation"]["latency_ms"].dropna() if len(e) else None
    hi = None
    for rr in _runs():
        dd = _read(rr, "decisions.csv")
        if dd.empty or "llm_reason" not in dd or "battery" not in dd:
            continue
        w = dd[dd["llm_reason"].astype(str).str.contains(
            "Battery warning: battery <= 15", na=False)]
        if len(w) and w["battery"].notna().any():
            v = float(w["battery"].max())
            hi = v if hi is None else max(hi, v)
    return {"records": len(s_), "calls": len(calls),
            "timeout_records": int(s_.str.contains("LLM error", na=False).sum()),
            "timeout_calls": int(calls.str.contains("LLM error", na=False).sum()),
            "timeout_pct_of_records": round(
                100 * s_.str.contains("LLM error", na=False).mean(), 1),
            "timeout_pct_of_calls": round(
                100 * calls.str.contains("LLM error", na=False).mean(), 1),
            "latency_mean_ms": round(float(lat.mean())) if lat is not None and len(lat) else None,
            "latency_p95_ms": round(float(np.percentile(lat, 95))) if lat is not None and len(lat) else None,
            "max_spurious_warning_charge_any_run": hi}


# ------------------------------------------------------------------- claims
# (section, claim as printed, value in the manuscript, derivation, reader)
CLAIMS = [
    ("abstract", "270 logged trials over 43 instrumented runs", (270, 43), d_counts,
     lambda v: (v["n_trials"], v["n_runs"])),
    ("abstract", "149 of them flown", 149, d_counts, lambda v: v["n_flight_trials"]),
    ("abstract", "ladder 0.997 / 0.910 / 0.783", (0.997, 0.910, 0.783),
     d_protocol_ladder,
     lambda v: (v["within_A"], v["loso_mean"], v["operational"])),
    ("abstract", "spread 0.214 between reported and deployed", 0.214,
     d_protocol_ladder, lambda v: v["gap_within_minus_operational"]),
    ("abstract", "background 0.784 against illumination 0.880", (0.784, 0.880),
     d_pair_factors, lambda v: (v["background_varies"], v["illumination_varies"])),
    ("III-B", "the original campaign ran the rule throughout",
     ["rule_based"], d_classifier_deployed,
     lambda v: v["manifest_classifier_values"]),

    ("V-D", "rule on pointing 0.584 over 2625 frames", (0.584, 2625), d_rule_operational,
     lambda v: (v["accuracy"], v["frames"])),
    ("V-D", "lateral F1 0.921 / 0.937 / 0.855", (0.921, 0.937, 0.855), d_rule_operational,
     lambda v: (v["f1"]["LEFT"], v["f1"]["RIGHT"], v["f1"]["DOWN"])),
    ("V-D", "depth F1 0.008 / 0.066", (0.008, 0.066), d_rule_operational,
     lambda v: (v["f1"]["FORWARD"], v["f1"]["BACK"])),
    ("V-D", "6 holds excluded, lateral inversion without exception", 6,
     d_rule_operational, lambda v: v["excluded_holds"]),
    ("V-D", "the excluded holds invert 211 of 211 and 216 of 216 frames",
     (211, 216), d_rule_operational,
     lambda v: (v["excluded_frames_left"], v["excluded_frames_right"])),
    ("V-F", "depth fires in every hold but 9% of the window", (1.0, 0.092),
     d_rule_operational, lambda v: (v["depth_hit_rate"], v["depth_frame_share"])),
    ("V-F", "67% of BACK holds emit a spurious FORWARD", 0.667, d_rule_operational,
     lambda v: v["back_opposite_rate"]),
    ("V-F", "gate changes accuracy by 0.004", 0.004, d_rule_operational,
     lambda v: v["gate_delta"]),
    # Generalisations over a table. Registered because the per-figure audit
    # cannot see them: the sentence asserts a count, not a number in the text.
    ("V-E", "no static class falls below 0.92 recall on the SVM", 5,
     d_generalisations, lambda v: v["svm_static_ge_092"]),
    ("V-E", "DOWN reaches 0.92 exactly, the SVM's lowest static class", 0.920,
     d_generalisations, lambda v: v["svm_static_min_recall"]),
    # The same sentence on F1. Stating it on recall while the table's caption
    # argues for F1 is how LEFT and RIGHT at 0.841 and 0.808 went unremarked.
    ("V-E", "on F1 two static classes fall short, LEFT and RIGHT", 3,
     d_generalisations, lambda v: v["svm_static_f1_ge_092"]),
    ("V-E", "two of five reach 0.92 on the rule", 2, d_generalisations,
     lambda v: v["rule_static_ge_092"]),
    ("V-E", "CENTER, UP and DOWN fall short on the rule", 3, d_generalisations,
     lambda v: v["rule_static_below_092"]),
    ("V-E", "depth contributes 82% of the 0.199 gap", 0.199, d_generalisations,
     lambda v: v["gap"]),
    # The headline must survive either weighting of the frame counts; the paper
    # does not apportion the remainder because it does not.
    ("V-E", "depth share 0.82 at each arm's own frame shares", 0.82,
     d_generalisations, lambda v: v["depth_share_own"]),
    ("V-E", "depth share 0.81 at pooled frame shares", 0.81,
     d_generalisations, lambda v: v["depth_share_pooled"]),
    ("V-E", "remainder 0.037 after the depth channel", 0.037,
     d_generalisations, lambda v: v["remainder_own"]),
    ("V-D", "a hand was detected in every frame of the ground SVM capture", 1.0,
     d_generalisations, lambda v: v["ground_svm_detection_rate"]),

    # ---- claims that must agree across sections -------------------------
    # V-J describes the six BACK holds, Table VII gives the per-frame recall and
    # V-E gives the split. A hold description implying a recall the table
    # contradicts is exactly the error these three checks exist to catch: the
    # manuscript carried "four of six read BACK" against a table recall of
    # 0.336 through a full revision.
    ("V-G", "per-hold BACK 0.333 against per-frame 0.336", (0.333, 0.336),
     d_cross_section, lambda v: (v["back_hold_recall"], v["back_frame_recall"])),
    ("V-G", "the two units agree on BACK to within 0.01", 0.0,
     d_cross_section, lambda v: v["back_hold_minus_frame"]),
    ("V-G", "the six BACK holds divide two BACK, two LEFT, two RIGHT", (2, 2, 2),
     d_cross_section, lambda v: (v["back_majority_counts"].get("BACK"),
                                 v["back_majority_counts"].get("LEFT"),
                                 v["back_majority_counts"].get("RIGHT"))),
    ("V-E", "BACK divides between LEFT at 0.33 and RIGHT at 0.30", (0.331, 0.298),
     d_cross_section, lambda v: (v["back_split"].get("LEFT"),
                                 v["back_split"].get("RIGHT"))),
    ("V-E", "the BACK split accounts for the whole class", 1.0,
     d_cross_section, lambda v: v["back_split_sums_to_one"]),
    # Face-following standoff. Registered after the published 1.7 m proved to
    # come from a 1280x720 assumption on a 960x720 stream and the superseded
    # 0.9--2.4 m envelope. Each of the three parts is checked: the code
    # constant, the measured calibration, and the conclusion drawn from them.
    # Table III binds Section III to Section V. Its left half is generated from
    # the deployed dataclasses; this fails if the .tex is hand-edited away from
    # them, which is how a specification silently stops describing the artefact.
    ("III", "every parameter in Table III matches the deployed configuration",
     (14, 14), d_design_table,
     lambda v: (v["params_matching_code"], v["params_checked"])),
    ("III", "Table III has one row per measured parameter", 15, d_design_table,
     lambda v: v["rows"]),
    ("III", "Table III carries the calibrated standoff, not a raw fraction",
     True, d_design_table, lambda v: v["standoff_row_present"]),

    ("III-D", "face-follow target area fraction 0.075", 0.075, d_follow_standoff,
     lambda v: v["target_area_frac"]),
    ("V-H", "sqrt(area) calibration 0.258 m over 4861 in-envelope frames",
     (0.258, 4861), d_follow_standoff,
     lambda v: (v["k_sqrt_area_m"], v["n_frames_in_envelope"])),
    ("V-H", "box-height calibration 214 px.m", 214.0, d_follow_standoff,
     lambda v: v["k_bbox_h_px_m"]),
    ("V-H", "target framing 0.94 m, band 0.90--1.00 m", (0.94, 0.90, 1.00),
     d_follow_standoff,
     lambda v: (v["standoff_m"], v["band_near_m"], v["band_far_m"])),
    ("V-H", "the target framing sits inside the 0.5--1.5 m envelope", True,
     d_follow_standoff, lambda v: v["inside_envelope"]),

    ("I", "Table I's headline cell is Table VII's per-frame SVM accuracy",
     0.850, d_cross_section, lambda v: v.get("table_I_cell")),
    ("V-D", "only two off-diagonal cells exceed 0.10, ground SVM", 2,
     d_generalisations, lambda v: v["ground_svm_offdiag_gt_010"]),
    ("V-C", "no off-diagonal cell exceeds 0.12 in folds C and D", 0.12,
     d_generalisations, lambda v: v["folds_CD_max_offdiag"]),
    ("V-F", "in-flight SVM 0.850 per frame, 0.905 per hold", (0.850, 0.905),
     d_inflight, lambda v: (v["svm"]["accuracy"], v["svm"]["hold_accuracy"])),
    ("V-F", "in-flight rule 0.651 per frame, 0.714 per hold", (0.651, 0.714),
     d_inflight, lambda v: (v["rule"]["accuracy"], v["rule"]["hold_accuracy"])),
    ("V-F", "in-flight depth: SVM 0.777 / 0.336 vs rule 0.000 / 0.008",
     (0.777, 0.336, 0.000, 0.008), d_inflight,
     lambda v: (v["svm"]["per_class"]["FORWARD"], v["svm"]["per_class"]["BACK"],
                v["rule"]["per_class"]["FORWARD"], v["rule"]["per_class"]["BACK"])),
    ("V-F", "in-flight detection above 0.98 both arms", (0.986, 0.984),
     d_inflight, lambda v: (v["svm"]["detection"], v["rule"]["detection"])),
    ("V-F", "42 holds per arm", (42, 42), d_inflight,
     lambda v: (v["svm"]["holds"], v["rule"]["holds"])),
    ("III-B", "zero runs carry the SVM latency column", 0, d_classifier_deployed,
     lambda v: v["runs_with_svm_latency_column"]),
    ("III-B", "16 runs carry the rule latency column", 16, d_classifier_deployed,
     lambda v: v["runs_with_rule_latency_column"]),
    ("III-B", "27 compound labels in flight", 27, d_classifier_deployed,
     lambda v: v["compound_labels_in_flight"]),

    ("contrib", "ladder quoted in the contributions bullet", (0.997, 0.910, 0.783),
     d_protocol_ladder, lambda v: (v["within_A"], v["loso_mean"], v["operational"])),

    ("concl", "B->D strongest pairing at 0.977", 0.977, d_transfer,
     lambda v: v["B->D"]),
    ("concl", "ladder quoted in the conclusion", (0.997, 0.910, 0.783),
     d_protocol_ladder, lambda v: (v["within_A"], v["loso_mean"], v["operational"])),

    ("V-L", "199 calls across 490 records", (199, 490), d_llm_denominators,
     lambda v: (v["calls"], v["records"])),
    ("V-L", "11 timeout calls, 5.5% of calls", (11, 5.5), d_llm_denominators,
     lambda v: (v["timeout_calls"], v["timeout_pct_of_calls"])),
    ("V-L", "35 timeout records, 7.1% of records", (35, 7.1), d_llm_denominators,
     lambda v: (v["timeout_records"], v["timeout_pct_of_records"])),
    ("V-L", "warning at charges up to 94%", 94, d_llm_denominators,
     lambda v: v["max_spurious_warning_charge_any_run"]),
    ("abstract", "6.2% of gesture frames carry an uncommanded depth component", 6.2,
     d_depth_contamination, lambda v: v["pct"]),
    ("abstract", "LOSO gesture accuracy 0.872", 0.872, d_loso, lambda v: v["mean"]),
    ("abstract", "within-session gesture accuracy 0.997", 0.997, d_within_session, lambda v: v["A"]),
    ("abstract", "in-flight face FRR 19.3%", 19.3, d_auth_rates,
     lambda v: v["drone_flight"]["frr_pct"]),

    ("IV-B", "mean grey 114 / 118 / 116 / 123", (114, 118, 116, 123), d_capture_conditions,
     lambda v: tuple(v[k]["grey"] for k in "ABCD")),
    ("IV-B", "blue-red 74 / 2 / 0 / -27", (74, 2, 0, -27), d_capture_conditions,
     lambda v: tuple(v[k]["blue_minus_red"] for k in "ABCD")),
    ("IV-B", "usable samples 2715 / 2345 / 2613 / 2784", (2715, 2345, 2613, 2784),
     d_loso, lambda v: tuple(v["test_n"][k] for k in "ABCD")),

    ("V-A", "149 valid in-flight trials, all passed", (149, 149), d_counts,
     lambda v: (v["n_flight_trials"], v["n_flight_passed"])),
    ("V-B", "median camera-to-command 18.6 ms", 18.6, d_runtime,
     lambda v: v["camera_to_command_ms"]["median"]),
    ("V-B", "p95 camera-to-command 55.2 ms", 55.2, d_runtime,
     lambda v: v["camera_to_command_ms"]["p95"]),
    ("V-B", "frame age median 4.0 ms, p95 30.5 ms", (4.0, 30.5), d_runtime,
     lambda v: (v["frame_age_ms"]["median"], v["frame_age_ms"]["p95"])),
    ("V-B", "22.3 fps over 557 s and 12,419 frames", (22.3, 557, 12419), d_runtime,
     lambda v: (v["fps"], v["airborne_s"], v["frames"])),

    ("V-C", "within-session 0.997 / 0.988 / 1.000", (0.997, 0.988, 1.0), d_within_session,
     lambda v: (v["A"], v["B"], v["C"])),
    ("V-C", "held-out 0.756 / 0.878 / 0.981", (0.756, 0.878, 0.981), d_loso,
     lambda v: (v["A"], v["B"], v["C"])),
    ("V-C", "sweep 0.9486 / 0.9840 / 0.9971", (0.9486, 0.984, 0.9971), d_sweep,
     lambda v: (v["100"], v["250"], v["400"])),
    ("V-C", "A->B 0.815, A->C 0.842", (0.815, 0.842), d_transfer,
     lambda v: (v["A->B"], v["A->C"])),
    ("V-C", "C->A 0.471, C->B 0.509", (0.471, 0.509), d_transfer,
     lambda v: (v["C->A"], v["C->B"])),
    ("V-C", "B->C 0.969", 0.969, d_transfer, lambda v: v["B->C"]),

    # Offline face benchmark. Produced by I. Chaabeni outside this repository,
    # so the audit attributes it rather than deriving it: EXTERNAL is not a
    # pass, it records who stands behind the number and that no log here
    # reproduces it.
    ("V-H", "offline EER 0.32% on 76 vs 13,410 images", 0.32, "EXTERNAL",
     "I. Chaabeni; offline still-image benchmark, generator not in this repo"),
    ("V-H", "live webcam FRR 1.1% (n=91)", (1.1, 91), d_auth_rates,
     lambda v: (v["webcam"]["frr_pct"], v["webcam"]["n"])),
    ("V-H", "drone in flight FRR 19.3% (n=684)", (19.3, 684), d_auth_rates,
     lambda v: (v["drone_flight"]["frr_pct"], v["drone_flight"]["n"])),
    # The unconditioned on-ground 32.0% over 122 frames was removed from the
    # manuscript as superseded, so it is no longer registered as a claim in it.
    # These are the two figures the paper now states in its place.
    ("V-H", "on-ground FRR in envelope 16.7% per frame, 9.2% gated, n=2792",
     (16.7, 9.2, 2792), d_ground_envelope_frr,
     lambda v: (v["frr_pct_per_frame"], v["frr_pct_gated"], v["n"])),

    ("V-I", "genuine cluster 0.770 (sd 0.109)", (0.770, 0.109), d_stale_crop,
     lambda v: (v["bench"]["genuine_mean"], v["bench"]["genuine_sd"])),
    ("V-I", "background cluster 0.193 (sd 0.033)", (0.193, 0.033), d_stale_crop,
     lambda v: (v["bench"]["background_mean"], v["bench"]["background_sd"])),
    ("V-I", "background share 56% -> 20%", (56, 20), d_stale_crop,
     lambda v: (v["bench"]["below_0.4_pct"], v["bench2"]["below_0.4_pct"])),
    ("V-I", "authorized share 41% -> 68%", (41, 68), d_stale_crop,
     lambda v: (round(v["bench"]["authorized_pct"]), round(v["bench2"]["authorized_pct"]))),
    ("V-I", "search_360 decisions 130 -> 4", (130, 4), d_stale_crop,
     lambda v: (v["bench"]["modes"].get("search_360"), v["bench2"]["modes"].get("search_360"))),
    ("V-I", "face decisions 72 -> 82", (72, 82), d_stale_crop,
     lambda v: (v["bench"]["modes"].get("face"), v["bench2"]["modes"].get("face"))),

    ("V-J", "mode-switch rate 15.7 / 78.5 / 9.7 per min", (15.7, 78.5, 9.7), d_ablation,
     lambda v: (v["deployed"], v["no_auth_hysteresis"], v["no_gate"])),
    ("V-J", "impostor authorization 0.0% -> 5.6%, 20 trials rejected", (0.0, 5.6, 20), d_impostor,
     lambda v: (v["bare"]["authorized_pct"], v["hysteresis"]["authorized_pct"],
                v["hysteresis"]["trials"])),
    ("V-J", "hand raised 2.08 s, preemption 4.20 s, median 1.53 s", (2.08, 4.20, 1.53),
     d_hysteresis_trial, lambda v: (v["hand_raised_at_s"], v["preempted_at_s"],
                                    v["median_preemption_s_successful_trials"])),
    ("V-J", "replay gives 13 of 15 authorized, not 5", (13, 15, 5), d_hysteresis_trial,
     lambda v: (v["authorized_under_replay"], v["frames"], v["authorized_as_logged"])),
    ("V-J", "replay preempts at 2.08 s", 2.08, d_hysteresis_trial,
     lambda v: v["replay_preemption_s"]),
    ("V-J", "score trace 0.675, 0.479, 0.507, 0.513, 0.423, 0.414, 0.440, 0.455",
     [0.675, 0.479, 0.507, 0.513, 0.423, 0.414, 0.440, 0.455], d_hysteresis_trial,
     lambda v: v["scores"][:8]),

    ("V-F", "64 active gesture frames in 22 trials, 4 contaminated (6.2%)", (64, 22, 4, 6.2),
     d_depth_contamination,
     lambda v: (v["active_frames"], v["trials"], v["contaminated"], v["pct"])),

    ("V-K", "search covers median 60 deg over 5 episodes", (60, 5), d_search,
     lambda v: (v["before"]["median_deg"], v["before"]["episodes"])),
    ("V-K", "retuned search 318 deg over 22 episodes, median 19.9 s", (318, 22, 19.9), d_search,
     lambda v: (v["after"]["median_deg"], v["after"]["episodes"], v["after"]["median_duration_s"])),
    ("V-K", "11 landings from 91% to 21% over 580 s", (11, 91, 21, 580), d_failsafe,
     lambda v: (v["landings"], v["battery_first"], v["battery_last"], v["airborne_s"])),
    ("V-K", "4-24 fps across 31 runs", (4, 24, 31), d_radio,
     lambda v: (math.floor(v["fps_min"]), math.ceil(v["fps_max"]), v["runs"])),
    ("V-K", "pooled r = -0.08 between fps and battery", -0.08, d_radio,
     lambda v: v["pearson_r_fps_battery"]),
    ("V-K", "8.4 fps with 48 reconnections", (8.4, 48), d_radio,
     lambda v: (round(v["most_reopens"]["fps"], 1), int(v["most_reopens"]["reopen"]))),

    ("V-L", "latency mean 1766 ms, p95 4002 ms", (1766, 4002), d_llm,
     lambda v: (v["latency_mean_ms"], v["latency_p95_ms"])),
]


def _close(a, b, tol=0.06):
    if a is None or b is None:
        return False
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))) or len(a) != len(b):
            return False
        return all(_close(x, y, tol) for x, y in zip(a, b))
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    return abs(a - b) <= max(tol * max(abs(a), abs(b)), tol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", default="outputs/metrics/claim_audit.json")
    args = ap.parse_args()

    cache, results = {}, []
    for section, text, stated, deriv, reader in CLAIMS:
        if deriv == "EXTERNAL":
            results.append({"section": section, "claim": text, "stated": stated,
                            "derived": None, "status": "EXTERNAL", "source": reader})
            continue
        if deriv is None or reader is None:
            results.append({"section": section, "claim": text, "stated": stated,
                            "derived": None, "status": "UNVERIFIABLE"})
            continue
        key = deriv.__name__
        if key not in cache:
            try:
                cache[key] = deriv()
            except Exception as e:
                cache[key] = None
                print(f"  [{key}] {type(e).__name__}: {e}")
        v = cache[key]
        try:
            got = reader(v) if v is not None else None
        except Exception:
            got = None
        status = "UNVERIFIABLE" if got is None else ("MATCH" if _close(stated, got) else "MISMATCH")
        results.append({"section": section, "claim": text, "stated": stated,
                        "derived": got, "status": status})

    w = max(len(r["claim"]) for r in results)
    print(f"\n{'sec':<9}{'claim':<{w + 2}}{'stated':<26}{'derived':<26}status")
    print("-" * (9 + w + 2 + 26 + 26 + 12))
    for r in results:
        print(f"{r['section']:<9}{r['claim']:<{w + 2}}{str(r['stated']):<26}"
              f"{str(r['derived']):<26}{r['status']}")

    # EXTERNAL is neither a pass nor a failure: the number is attributed to a
    # named source outside this repository, so it is excluded from the derived
    # tally and listed with its provenance rather than silently counted.
    ext = [r for r in results if r["status"] == "EXTERNAL"]
    bad = [r for r in results if r["status"] not in ("MATCH", "EXTERNAL")]
    n_derivable = len(results) - len(ext)
    print(f"\n{n_derivable - len(bad)}/{n_derivable} log-derived claims reproduce.")
    for r in ext:
        print(f"  EXTERNAL      {r['section']:<8}{r['claim']}")
        print(f"                source: {r['source']}")
    for r in bad:
        print(f"  {r['status']:<13} {r['section']:<8}{r['claim']}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"results": results, "derived": cache}, f, indent=2, default=str)
    print(f"\nwrote {args.out_json}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
