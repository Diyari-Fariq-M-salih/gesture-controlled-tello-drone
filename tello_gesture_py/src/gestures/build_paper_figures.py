"""Regenerate every data figure in the manuscript directly from the run logs.

    python -m tello_gesture_py.src.gestures.build_paper_figures --outdir paper/figures

Palette is the validated categorical default (checks: chroma floor, normal-vision
separation, protan/deutan separation, contrast vs surface). Series identity is
never carried by colour alone -- every figure also uses a distinct marker, dash
pattern or direct label, so the figures survive greyscale printing.
"""

from tello_gesture_py.src.gestures.evidence import paper_glob
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical order
INK, MUTED, GRID = "#1a1a19", "#5b6573", "#dfe4ea"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

COL1 = 3.4   # IEEE single column, inches
COL2 = 7.1


def tidy(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def read(run, name, exact=True):
    """Load a log by run suffix. `exact` avoids the trap that a glob for
    'bench' also matches 'bench2'."""
    # Run directories are "<YYYYMMDD-HHMMSS>_<run-id>"; match the id exactly.
    cands = []
    for d in paper_glob("outputs/runs/*/"):
        base = os.path.basename(os.path.normpath(d))
        rid = base.split("_", 1)[1] if "_" in base else base
        if (rid == run) if exact else (run in rid):
            f = os.path.join(d, name)
            if os.path.exists(f):
                cands.append(f)
    if not cands:
        return pd.DataFrame()
    # Of equally-named runs keep the one with the most rows: repeated launches
    # of the same id leave short aborted directories behind.
    best, best_n = pd.DataFrame(), -1
    for f in sorted(cands):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df) > best_n:
            best, best_n = df, len(df)
    return best


def read_richest(run, name):
    """Of all runs matching a suffix, the one with the most rows."""
    best = pd.DataFrame()
    for d in paper_glob("outputs/runs/*/"):
        if run not in os.path.basename(os.path.normpath(d)):
            continue
        f = os.path.join(d, name)
        if not os.path.exists(f):
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if len(df) > len(best):
            best = df
    return best


# ---------------------------------------------------------------- fig: identity
def fig_score_distribution(outdir):
    """Bimodal before the stale-crop fix, unimodal after."""
    before, after = read("drone_bench", "decisions.csv"), read("drone_bench-2", "decisions.csv")
    assert not before.equals(after), "before/after resolved to the same run"
    if before.empty or after.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.0), sharey=True)
    # percentage of frames, so panels with 201 and 122 samples compare
    bins = np.linspace(0, 1, 34)
    for ax, d, title, colour in ((axes[0], before, "(a) Before: stale crops admitted", C2),
                                 (axes[1], after, "(b) After: crops require a fresh detection", C1)):
        s = d[(d["faceid_enrolled"] == True) & (d["faceid_score"] > -1)]["faceid_score"]  # noqa: E712
        w = np.ones(len(s)) * 100.0 / max(len(s), 1)
        ax.hist(s, bins=bins, weights=w, color=colour, edgecolor="white",
                linewidth=0.4, zorder=2)
        ax.axvline(0.55, color=INK, linewidth=1.0, linestyle="--", zorder=3)
        # Statistics go in the title: inside the axes they collide with the
        # tallest bar, which is the very feature the figure exists to show.
        ax.set_title("%s\nbelow 0.4: %.0f%%    authorized: %.0f%%"
                     % (title, 100 * (s < 0.4).mean(), 100 * (s >= 0.55).mean()),
                     loc="left", fontsize=7)
        ax.set_xlabel("cosine similarity to enrolled template")
        ax.set_xlim(0, 1)
        tidy(ax)
        # inside the axes, in the empty band just right of the threshold
        ax.annotate(r"$\tau=0.55$", xy=(0.575, 0.70),
                    xycoords=("data", "axes fraction"),
                    ha="left", va="center", fontsize=6.5, color=INK)
    axes[0].set_ylabel("frames (%)")
    p = os.path.join(outdir, "fig_score_distribution.pdf")
    fig.savefig(p); plt.close(fig)
    return p


def fig_envelope(outdir):
    """Authorization rate against apparent face size, with Wilson intervals."""
    import math

    def wilson(k, n, z=1.96):
        if n == 0:
            return 0.0, 0.0
        p = k / n
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return max(0.0, c - h), min(1.0, c + h)

    d = read("drone_distance", "decisions.csv")
    if d.empty:
        return None
    d = d[(d["faceid_enrolled"] == True) & (d["face_raw"] == True) & d["face_bbox_h"].notna()]  # noqa: E712
    bins = [0, 150, 180, 220, 280, 350, 480, 620]
    xs, rates, los, his, ns = [], [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        g = d[(d["face_bbox_h"] >= lo) & (d["face_bbox_h"] < hi)]
        if len(g) < 2:
            continue
        k, n = int(g["face_auth"].sum()), len(g)
        a, b = wilson(k, n)
        xs.append((lo + hi) / 2)
        rates.append(100 * k / n)
        los.append(100 * (k / n - a))
        his.append(100 * (b - k / n))
        ns.append(n)

    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    ax.axvspan(180, 480, color=C1, alpha=0.09, zorder=1)
    ax.errorbar(xs, rates, yerr=[los, his], fmt="o", color=C1, markersize=5,
                markerfacecolor="white", markeredgewidth=1.5, capsize=2.5,
                elinewidth=1.0, zorder=3)
    # counts sit above each marker, clear of both the bars and the tick labels
    for x, r, h, n in zip(xs, rates, his, ns):
        ax.annotate("n=%d" % n, (x, min(r + h + 4, 104)), ha="center",
                    fontsize=5.8, color=MUTED)
    ax.annotate("usable envelope", (330, 116), ha="center", fontsize=7, color=C1,
                annotation_clip=False)
    ax.annotate("", xy=(180, 110), xytext=(480, 110),
                arrowprops=dict(arrowstyle="<->", color=C1, lw=0.9),
                annotation_clip=False)
    ax.set_xlabel("face bounding-box height (px)")
    ax.set_ylabel("frames authorized (%)")
    ax.set_ylim(-6, 112)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_yticks([0, 25, 50, 75, 100])
    tidy(ax)
    p = os.path.join(outdir, "fig_envelope.pdf")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- fig: failsafe
def fig_failsafe(outdir):
    """Battery trace with every automatic landing marked."""
    t, e = read("drone_n4-failsafe-ladder", "telemetry.csv"), read("drone_n4-failsafe-ladder", "perf_events.csv")
    if t.empty:
        return None
    t0 = t["t"].min()
    fig, ax = plt.subplots(figsize=(COL2, 2.2))
    ax.plot(t["t"] - t0, t["bat"], color=C1, linewidth=1.6, zorder=3, label="battery")
    ax2 = ax.twiny() if False else None  # single axis by design
    fs = e[e["event"] == "battery_failsafe_land"] if not e.empty else pd.DataFrame()
    for i, (_, r) in enumerate(fs.iterrows()):
        near = t.iloc[(t["t"] - r["t"]).abs().argmin()]
        ax.plot(r["t"] - t0, near["bat"], marker="v", markersize=7, color=C2,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5,
                label="automatic landing" if i == 0 else None)
    ax.set_xlabel("time since launch (s)")
    ax.set_ylabel("battery (%)")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.02, 0.06, f"{len(fs)} consecutive failsafes, 7-point steps",
            transform=ax.transAxes, fontsize=7, color=MUTED)
    tidy(ax)
    p = os.path.join(outdir, "fig_failsafe.pdf")
    fig.savefig(p); plt.close(fig)
    return p


# ---------------------------------------------------------------- fig: search
def fig_search(outdir):
    """Sweep coverage per episode, as specified vs as retuned."""
    def sweeps(pat):
        t = read_richest(pat, "telemetry.csv")
        d = read_richest(pat, "decisions.csv")
        if t.empty or d.empty:
            return []
        d = d.sort_values("t")
        eps, cur = [], None
        for _, r in d.iterrows():
            if r["mode"] == "search_360":
                cur = [r["t"], r["t"]] if cur is None else [cur[0], r["t"]]
            elif cur is not None:
                eps.append(cur); cur = None
        out = []
        for a, b in eps:
            sub = t[(t["t"] >= a) & (t["t"] <= b)]
            if len(sub) < 3 or (b - a) < 2.0 or not (sub["h"] > 0).any():
                continue
            y = sub["yaw"].values.astype(float)
            out.append(abs(float(np.sum((np.diff(y) + 180) % 360 - 180))))
        return out
    old, new = sweeps("drone_home-flight-1"), sweeps("drone_n1-scenarios-6-7")
    if not old or not new:
        return None
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    parts = ax.boxplot([old, new], widths=0.5, patch_artist=True,
                       medianprops=dict(color=INK, linewidth=1.4),
                       flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED,
                                       markeredgecolor="none", alpha=0.6))
    for patch, colour in zip(parts["boxes"], (C2, C1)):
        patch.set_facecolor(colour); patch.set_alpha(0.65); patch.set_edgecolor(colour)
    for whisk in parts["whiskers"] + parts["caps"]:
        whisk.set_color(MUTED); whisk.set_linewidth(0.8)
    ax.axhline(360, color=INK, linestyle="--", linewidth=1.0, zorder=1)
    ax.text(2.45, 368, "full rotation", fontsize=7, color=INK, ha="right")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"5 s\n(as specified)\nn={len(old)}", f"28 s\n(retuned)\nn={len(new)}"])
    ax.set_ylabel("yaw swept per episode (deg)")
    ax.set_ylim(0, 420)
    tidy(ax)
    p = os.path.join(outdir, "fig_search.pdf")
    fig.savefig(p); plt.close(fig)
    return p


# ------------------------------------------------------------ fig: crosssession
def fig_crosssession(outdir):
    """Per-class F1 across the three session pairings."""
    names = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN", "FORWARD", "BACK"]
    data = {}
    for k in ("AB", "AC", "BC"):
        f = f"outputs/metrics/cross_session_{k}.json"
        if os.path.exists(f):
            data[k] = json.load(open(f, encoding="utf-8"))
    if len(data) < 3:
        return None
    fig, ax = plt.subplots(figsize=(COL2, 2.3))
    x = np.arange(len(names)); w = 0.26
    series = [("A$\\rightarrow$B", "AB", C1, "//"), ("A$\\rightarrow$C", "AC", C2, "\\\\"),
              ("B$\\rightarrow$C", "BC", C3, "")]
    for i, (label, key, colour, hatch) in enumerate(series):
        vals = [data[key]["report"].get(str(j), {}).get("f1-score", 0) for j in range(7)]
        ax.bar(x + (i - 1) * w, vals, w * 0.88, label=label, color=colour,
               edgecolor="white", linewidth=0.6, hatch=hatch, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("per-class $F_1$"); ax.set_ylim(0, 1.09)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    tidy(ax)
    p = os.path.join(outdir, "fig_crosssession.pdf")
    fig.savefig(p); plt.close(fig)
    return p


# ------------------------------------------------------------------ fig: link
def fig_link(outdir):
    """Stream frame rate against battery state of charge."""
    pts = []
    for d in sorted(paper_glob("outputs/runs/*/")):
        try:
            t = pd.read_csv(os.path.join(d, "telemetry.csv"))
            p = pd.read_csv(os.path.join(d, "perf.csv"))
        except Exception:
            continue
        if len(p) < 200 or len(t) < 5 or "bat" not in t:
            continue
        span = p["t"].max() - p["t"].min()
        if span < 20:
            continue
        venue = "campus" if os.path.basename(os.path.normpath(d))[:8] == "20260828" else "home"
        pts.append((float(t["bat"].iloc[0]), (len(p) - 1) / span, venue))
    if len(pts) < 4:
        return None
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    # Venue dominates: pooling the two sites hides that the congested site is
    # slow at every charge level. Plot them as separate series.
    for (venue, colour, marker) in (("home", C1, "o"), ("campus", C2, "^")):
        g = [x for x in pts if x[2] == venue]
        if not g:
            continue
        b = np.array([x[0] for x in g]); f = np.array([x[1] for x in g])
        r = float(np.corrcoef(b, f)[0, 1]) if len(g) > 2 else float("nan")
        ax.scatter(b, f, s=26, color=colour, marker=marker, edgecolor="white",
                   linewidth=0.7, zorder=3, label=f"{venue} ($r={r:.2f}$, $n={len(g)}$)")
    ax.legend(frameon=False, loc="lower right", fontsize=6.8)
    ax.set_xlabel("battery at launch (%)")
    ax.set_ylabel("mean stream rate (fps)")
    tidy(ax)
    p = os.path.join(outdir, "fig_link.pdf")
    fig.savefig(p); plt.close(fig)
    return p


def fig_confusion(outdir):
    """Row of three confusion matrices, one per session pairing."""
    names = ["CEN", "LEF", "RIG", "UP", "DWN", "FWD", "BCK"]
    pairs = [("AB", r"(a) A$\rightarrow$B"), ("AC", r"(b) A$\rightarrow$C"),
             ("BC", r"(c) B$\rightarrow$C")]
    data = {}
    for k, _ in pairs:
        f = f"outputs/metrics/cross_session_{k}.json"
        if os.path.exists(f):
            data[k] = json.load(open(f, encoding="utf-8"))
    if len(data) < 3:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))
    for ax, (key, title) in zip(axes, pairs):
        cm = np.array(data[key]["confusion_matrix"], dtype=float)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(7)); ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticks(range(7)); ax.set_yticklabels(names if key == "AB" else [], fontsize=6)
        ax.set_title(f"{title}  acc {data[key]['accuracy']:.3f}", fontsize=7.5)
        ax.set_xlabel("predicted", fontsize=7)
        if key == "AB":
            ax.set_ylabel("true", fontsize=7)
        for i in range(7):
            for j in range(7):
                v = cmn[i, j]
                if v >= 0.01:
                    ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                            fontsize=5.2, color="white" if v > 0.55 else INK)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.016, pad=0.012)
    cb.set_label("row-normalised rate", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)
    p = os.path.join(outdir, "fig_confusion.pdf")
    fig.savefig(p); plt.close(fig)
    return p


def fig_loso(outdir):
    """Leave-one-session-out confusion matrices, one per held-out session."""
    names = ["CEN", "LEF", "RIG", "UP", "DWN", "FWD", "BCK"]
    f = "outputs/metrics/loso_eval.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f, encoding="utf-8"))["loso"]

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))
    for ax, held in zip(axes, ("A", "B", "C")):
        r = d[held]
        cm = np.array(r["confusion_matrix"], dtype=float)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(7)); ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticks(range(7))
        ax.set_yticklabels(names if held == "A" else [], fontsize=6)
        tr = "+".join(r["train_sessions"])
        idx = ("A", "B", "C").index(held)
        ax.set_title("(%s) test %s, train %s   acc %.3f" % ("abc"[idx], held, tr, r["accuracy"]),
                     fontsize=7.5)
        ax.set_xlabel("predicted", fontsize=7)
        if held == "A":
            ax.set_ylabel("true", fontsize=7)
        for i in range(7):
            for j in range(7):
                v = cmn[i, j]
                if v >= 0.01:
                    ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                            fontsize=5.2, color="white" if v > 0.55 else INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.016, pad=0.012)
    cb.set_label("row-normalised rate", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)
    p_ = os.path.join(outdir, "fig_loso.pdf")
    fig.savefig(p_); plt.close(fig)
    return p_


def fig_per_session(outdir):
    """Within-session confusion matrix for each session, trained and tested on itself."""
    names = ["CEN", "LEF", "RIG", "UP", "DWN", "FWD", "BCK"]
    f = "outputs/metrics/per_session_eval.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f, encoding="utf-8"))

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))
    for ax, key, tag in zip(axes, ("A", "B", "C"), ("a", "b", "c")):
        r = d[key]
        cm = np.array(r["confusion_matrix"], dtype=float)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(7)); ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticks(range(7))
        ax.set_yticklabels(names if key == "A" else [], fontsize=6)
        ax.set_title("(%s) session %s   acc %.3f" % (tag, key, r["accuracy"]), fontsize=7.5)
        ax.set_xlabel("predicted", fontsize=7)
        if key == "A":
            ax.set_ylabel("true", fontsize=7)
        for i in range(7):
            for j in range(7):
                v = cmn[i, j]
                if v >= 0.01:
                    ax.text(j, i, ("%.2f" % v).lstrip("0"), ha="center", va="center",
                            fontsize=5.2, color="white" if v > 0.55 else INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.016, pad=0.012)
    cb.set_label("row-normalised rate", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)
    p_ = os.path.join(outdir, "fig_per_session.pdf")
    fig.savefig(p_); plt.close(fig)
    return p_


def fig_sessions(outdir):
    """Leave-one-session-out confusion, one panel per held-out session.

    This carried a within-session row as well. It was dropped: those four
    accuracies are quoted in the text, and the section's argument is that they
    are uninformative, so the row illustrated a non-result at the cost of half
    a figure. What the held-out row shows -- which classes actually collapse,
    and into what -- is the part the prose reasons about.
    """
    names = ["CEN", "LEF", "RIG", "UP", "DWN", "FWD", "BCK"]
    fa = "outputs/metrics/per_session_eval.json"
    fb = "outputs/metrics/loso_eval.json"
    if not (os.path.exists(fa) and os.path.exists(fb)):
        return None
    within = json.load(open(fa, encoding="utf-8"))
    loso = json.load(open(fb, encoding="utf-8"))["loso"]
    # Kept only to fix the panel set to the sessions both evaluations cover.

    # However many sessions the evaluations found. Adding a session should
    # widen this figure, not require editing it.
    keys = [k for k in sorted(set(within) & set(loso))]
    ncol = len(keys)
    fig, axes = plt.subplots(1, ncol, figsize=(COL2 * max(ncol, 3) / 3.0, 2.35),
                             squeeze=False)
    for row, (src, prefix) in enumerate(((loso, "held out"),)):
        for col, key in enumerate(keys):
            ax = axes[row][col]
            r = src[key]
            cm = np.array(r["confusion_matrix"], dtype=float)
            cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(7))
            ax.set_xticklabels(names, rotation=90, fontsize=6)
            ax.set_yticks(range(7))
            ax.set_yticklabels(names if col == 0 else [], fontsize=6)
            tag = "abcdefghij"[col]
            ax.set_title("(%s) session %s held out   %.3f" % (tag, key, r["accuracy"]),
                         fontsize=7)
            ax.set_xlabel("predicted", fontsize=7)
            if col == 0:
                ax.set_ylabel("true", fontsize=7)
            for i in range(7):
                for j in range(7):
                    v = cmn[i, j]
                    if v >= 0.01:
                        ax.text(j, i, ("%.2f" % v).lstrip("0"), ha="center", va="center",
                                fontsize=4.8, color="white" if v > 0.55 else INK)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes, fraction=0.016, pad=0.012)
    cb.set_label("row-normalised rate", fontsize=7)
    cb.ax.tick_params(labelsize=6, length=2)
    p_ = os.path.join(outdir, "fig_sessions.pdf")
    fig.savefig(p_); plt.close(fig)
    return p_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    # fig_envelope is deliberately absent. It bins by apparent face size, a
    # proxy whose extreme bins held two and three frames, and it was superseded
    # by fig_envelope_distance, which draws the same envelope against declared
    # standoff at flight-free n and writes to the same path. Running it here
    # silently clobbered Figure 2 once; it will not again.
    for fn in (fig_score_distribution, fig_failsafe,
               fig_search, fig_crosssession, fig_confusion, fig_loso, fig_per_session, fig_sessions, fig_link):
        try:
            p = fn(args.outdir)
            print(f"  {'wrote ' + p if p else 'SKIPPED ' + fn.__name__ + ' (no data)'}")
        except Exception as e:
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
