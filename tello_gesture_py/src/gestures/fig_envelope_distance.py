"""Figure 2, rebuilt on declared standoff distance instead of bounding-box bins.

The published figure bins by apparent face size, which is a proxy the reader
cannot act on, and its extreme bins held two and three frames. The sweep declares
distance directly, so the envelope can be drawn against the quantity a deployment
actually sets, at flight-free sample sizes.
"""
from tello_gesture_py.src.gestures.evidence import paper_glob
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from tello_gesture_py.src.gestures.build_paper_figures import (
    COL1, INK, MUTED, C1, C2, plt, tidy)

TAU_DEPLOY = 0.55
TAU_OFFLINE = 0.1816     # EER-optimal point from the offline benchmark


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def main():
    d = pd.concat([pd.read_csv(p) for p in
                   paper_glob("outputs/runs/*/stream_identity.csv")], ignore_index=True)
    d = d[d["faceid_score"].notna()]

    xs, rate, lo, hi, ns, med = [], [], [], [], [], []
    for v, g in d.groupby("distance_m"):
        k, n = int(g["face_auth"].sum()), len(g)
        a, b = wilson(k, n)
        xs.append(v)
        rate.append(100 * k / n)
        lo.append(100 * (k / n - a))
        hi.append(100 * (b - k / n))
        ns.append(n)
        med.append(float(g["faceid_score"].median()))

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(COL1, 3.5), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1.15]})

    # usable band: contiguous marks at or above 75 %
    ok = [x for x, r in zip(xs, rate) if r >= 75]
    if ok:
        ax.axvspan(min(ok) - 0.12, max(ok) + 0.12, color=C1, alpha=0.09, zorder=1)
    ax.errorbar(xs, rate, yerr=[lo, hi], fmt="o-", color=C1, markersize=4.5,
                markerfacecolor="white", markeredgewidth=1.3, capsize=2.5,
                elinewidth=1.0, linewidth=1.0, zorder=3)
    for x, r, h, n in zip(xs, rate, hi, ns):
        ax.annotate("%d" % n, (x, min(r + h + 5, 103)), ha="center", fontsize=5.6,
                    color=MUTED)
    ax.set_ylabel("frames authorized (%)")
    ax.set_ylim(-6, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.annotate("usable envelope", ((min(ok) + max(ok)) / 2 if ok else 1.0, 112),
                ha="center", fontsize=7, color=C1, annotation_clip=False)
    tidy(ax)

    ax2.plot(xs, med, "s-", color=C2, markersize=4, markerfacecolor="white",
             markeredgewidth=1.2, linewidth=1.0, zorder=3)
    ax2.axhline(TAU_DEPLOY, color=INK, linewidth=0.9, linestyle="--", zorder=2)
    ax2.axhline(TAU_OFFLINE, color=MUTED, linewidth=0.9, linestyle=":", zorder=2)
    # Both labels sit on the left, where the score curve runs high and clear of
    # them. Right-aligning them put both on top of the 2.5 and 3.0 m markers, which
    # is where the curve returns to the offline threshold. The white background
    # masks the rule underneath so the text reads at print size.
    _box = dict(facecolor="white", edgecolor="none", pad=0.8)
    ax2.annotate(r"$\tau_{\mathrm{on}}=0.55$ deployed",
                 (min(xs) - 0.12, TAU_DEPLOY - 0.055), ha="left", va="top",
                 fontsize=6.4, color=INK, bbox=_box, zorder=4)
    ax2.annotate(r"$\tau=0.18$ offline EER point",
                 (min(xs) - 0.12, TAU_OFFLINE - 0.045), ha="left", va="top",
                 fontsize=6.4, color=MUTED, bbox=_box, zorder=4)
    ax2.set_xlabel("operator standoff (m)")
    ax2.set_ylabel("median score")
    ax2.set_ylim(0, 1.0)
    ax2.set_xlim(min(xs) - 0.2, max(xs) + 0.2)
    tidy(ax2)

    p = "paper/figures/fig_envelope.pdf"
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)
    for x, r, n, m in zip(xs, rate, ns, med):
        print(f"   {x:4.2f} m  {r:5.1f}%  n={n:4d}  median score {m:.3f}")


if __name__ == "__main__":
    main()
