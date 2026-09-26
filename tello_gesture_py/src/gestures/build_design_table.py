"""The specification/measurement table binding Section III to Section V.

One row per design parameter that the results section measures. The left half is
read from the deployed dataclasses, so the table cannot drift from the code; the
right half is the figure the corresponding results subsection reports.

The point the table makes is visible in its own ordering: the parameters set by
reasoning rather than by measurement are the ones the flight campaign moved.

    python -m tello_gesture_py.src.gestures.build_design_table
"""

from tello_gesture_py.src.gestures.evidence import paper_glob
import argparse
import io
import json
import math
import os

import numpy as np
import pandas as pd

from ..config import ControllerConfig
from ..face_follow import FaceFollowConfig


def follow_standoff(cfg):
    """Target area fraction -> metres, calibrated on the envelope sweep."""
    import glob
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
    area = fit["face_bbox_w"] * fit["face_bbox_h"] / float(960 * 720)
    k = float(np.median(np.sqrt(area) * fit["distance_m"]))
    t, dd = cfg.target_area_frac, cfg.deadband_area
    return (k / math.sqrt(t), k / math.sqrt(t + dd), k / math.sqrt(t - dd))


def build():
    c, f = ControllerConfig(), FaceFollowConfig()
    st = follow_standoff(f)
    band = ("%.2f~m, band %.2f--%.2f~m" % st) if st else "--"

    # (parameter, value, set in, measured in, result)
    rows = [
        (r"Hand, face throttle", f"{c.hand_every_n}, {c.face_every_n} frames",
         r"\ref{sec:perception}", r"\ref{sec:runtime}",
         r"duty 50\%, 25\%; 22.3~fps"),
        (r"EMA $\alpha$ (rule only)", f"{c.ema_alpha}",
         r"\ref{sec:perception}", r"\ref{sec:inflight}",
         r"arms not input-matched"),
        (r"$\theta_{\mathrm{dir}}$", f"{c.dir_thr}",
         r"\ref{sec:vocab}", r"\ref{sec:opgesture}",
         r"lateral $F_1$ 0.921 / 0.937 / 0.855"),
        (r"$\theta_{\mathrm{scale}}$ on $\Delta s_t$", f"{c.scale_thr}",
         r"\ref{sec:vocab}", r"\ref{sec:depth}",
         r"depth $F_1$ 0.008 / 0.066; pose form 0.447 / 0.567"),
        (r"Depth stability streak", f"{c.gesture_fb_streak_on} frames",
         r"\ref{sec:vocab}", r"\ref{sec:depth}",
         r"gated vs ungated differ by 0.004"),
        (r"Face-follow target $\rho^{\star}$", f"{f.target_area_frac} of frame",
         r"\ref{sec:facefollow}", r"\ref{sec:identity}", band),
        (r"$\tau_{\mathrm{on}}$", f"{c.faceid_cosine_thr}",
         r"\ref{sec:gate}", r"\ref{sec:identity}",
         r"FRR 19.3\% in flight; 35.1\% ungated at the same threshold"),
        (r"$\tau_{\mathrm{off}}$, $k$",
         f"{c.faceid_release_thr}, {c.faceid_release_frames}",
         r"\ref{sec:gate}", r"\ref{sec:hysteresis}",
         r"15.7 vs 78.5 mode switches/min; impostor 0.0\% $\to$ 5.6\%"),
        (r"Enrolment $N$", f"{c.faceid_enroll_samples} crops",
         r"\ref{sec:gate}", r"\ref{sec:identity}",
         r"genuine cluster $0.770\pm0.109$"),
        (r"$\Delta_{\max}$ crop freshness", f"{f.crop_max_age_s}~s",
         r"\ref{sec:gate}", r"\ref{sec:staleness}",
         r"background share 56\% $\to$ 20\%; authorized 41\% $\to$ 68\%"),
        (r"Arbitration face hold", "2.0~s",
         r"\ref{sec:gate}", r"\ref{sec:staleness}",
         r"search decisions 130 $\to$ 4"),
        (r"Mode hold, release",
         f"{c.mode_hold_s}~s, {c.hand_release_s}~s",
         r"\ref{sec:arb}", r"\ref{sec:results}",
         r"149/149 trials produced the specified behaviour"),
        (r"Search sweep", f"{c.search_duration_s:.0f}~s",
         r"\ref{sec:arb}", r"\ref{sec:flightbehaviour}",
         r"318\textdegree{} over 22 episodes; 60\textdegree{} at the 5~s "
         r"originally specified"),
        (r"Battery failsafe", f"{c.battery_land_pct}\\%",
         r"\ref{sec:arb}", r"\ref{sec:flightbehaviour}",
         r"11 consecutive landings, 91\% $\to$ 21\%"),
        (r"LLM decision rate", f"{c.llm_decision_hz:.0f}~Hz, no actuation",
         r"\ref{sec:reasondesign}", r"\ref{sec:llm}",
         r"1766~ms mean, 5.5\% timeouts"),
    ]

    L = [r"\begin{table*}[t]",
         r"\caption{Parameters Section~\ref{sec:system} specifies and "
         r"Section~\ref{sec:results} measures. The left half is generated from "
         r"the deployed configuration.}",
         r"\label{tab:design}", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.05}",
         r"\begin{tabular}{@{}llccl@{}}", r"\toprule",
         r"Parameter & Deployed value & Set in & Measured in & Result \\",
         r"\midrule"]
    for p, v, s, m, r in rows:
        L.append(f"{p} & {v} & \\S\\,{s} & \\S\\,{m} & {r} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    out = "paper/tables/tab_design.tex"
    io.open(out, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    return out, rows


# The short paper has five results subsections rather than twelve, so a row can
# only point at one that exists there. Keyed by the row's parameter label.
SHORT_ROWS = {
    r"$\theta_{\mathrm{scale}}$ on $\Delta s_t$": "sec:behaviours",
    r"Face-follow target $\rho^{\star}$": "sec:identity",
    r"$\tau_{\mathrm{on}}$": "sec:identity",
    r"$\tau_{\mathrm{off}}$, $k$": "sec:behaviours",
    r"$\Delta_{\max}$ crop freshness": "sec:behaviours",
    r"Mode hold, release": "sec:results",
    r"Search sweep": "sec:behaviours",
    r"Battery failsafe": "sec:behaviours",
}


def build_short():
    """The same table for the short paper, pointing only at sections it has."""
    _, rows = build()
    keep = [r for r in rows if r[0] in SHORT_ROWS]
    L = [r"\begin{table}[t]",
         r"\caption{Parameters Section~\ref{sec:method} specifies and "
         r"Section~\ref{sec:results} measures. Values are generated from the "
         r"deployed configuration.}",
         r"\label{tab:design}", r"\centering", r"\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\renewcommand{\arraystretch}{1.05}",
         r"\begin{tabular}{@{}llc@{}}", r"\toprule",
         r"Parameter & Value & Measured in \\", r"\midrule"]
    for p, v, _s, _m, _r in keep:
        L.append("%s & %s & \\S\\,\\ref{%s} \\\\" % (p, v, SHORT_ROWS[p]))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = "paper/tables/tab_design_short.tex"
    io.open(out, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    return out, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", action="store_true")
    a = ap.parse_args()
    out, rows = build_short() if a.short else build()
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
