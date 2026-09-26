# tello_gesture/utils/reporting.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save_line_plot(df: pd.DataFrame, x: str, ys: list[str], title: str, outpath: Path):
    plt.figure()
    for y in ys:
        if y in df.columns:
            plt.plot(df[x].values, df[y].values, label=y)
    plt.title(title)
    plt.xlabel(x)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def _save_bar_plot(series: pd.Series, title: str, xlabel: str, ylabel: str, outpath: Path):
    plt.figure()
    series.sort_index().plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def telemetry_report(telemetry_csv: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(telemetry_csv)

    # If timestamps are ms-ish, make a seconds axis
    if "t" in df.columns and np.issubdtype(df["t"].dtype, np.number):
        t0 = float(df["t"].iloc[0])
        df["t_sec"] = (df["t"].astype(float) - t0) / 1000.0
        x = "t_sec"
    else:
        x = df.columns[0]

    # Plots based on your header style: t,bat,h,tof,yaw,vgx,vgy,vgz ...
    if "bat" in df.columns:
        _save_line_plot(df, x, ["bat"], "Battery (%)", outdir / "battery.png")
    if "h" in df.columns:
        _save_line_plot(df, x, ["h"], "Height (h)", outdir / "height.png")
    if "tof" in df.columns:
        _save_line_plot(df, x, ["tof"], "ToF (tof)", outdir / "tof.png")
    if "yaw" in df.columns:
        _save_line_plot(df, x, ["yaw"], "Yaw", outdir / "yaw.png")

    vel = [c for c in ["vgx", "vgy", "vgz"] if c in df.columns]
    if vel:
        _save_line_plot(df, x, vel, "Velocity (vgx/vgy/vgz)", outdir / "velocity.png")

    # Export a cleaned xlsx for sharing
    df.to_excel(outdir / "telemetry_clean.xlsx", index=False)


def dataset_image_report(dataset_csv: Path, labels_json: Optional[Path], outdir: Path):
    """
    dataset.csv written by auto_collect_dataset.py is: path,label :contentReference[oaicite:3]{index=3}
    """
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset_csv)

    # normalize columns
    if "path" not in df.columns and len(df.columns) >= 2:
        df = df.rename(columns={df.columns[0]: "path", df.columns[1]: "label"})

    df["label"] = df["label"].astype(int)

    label_map = None
    if labels_json and labels_json.exists():
        data = json.loads(labels_json.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "labels" in data:
            label_map = {int(x["id"]): str(x["name"]) for x in data["labels"]}
        elif isinstance(data, dict):
            label_map = {int(k): str(v) for k, v in data.items()}

    counts = df["label"].value_counts().sort_index()
    if label_map:
        counts.index = [f"{i:02d}_{label_map.get(int(i), str(i))}" for i in counts.index]

    _save_bar_plot(counts, "Images per class", "Class", "Count", outdir / "images_per_class.png")
    df.to_csv(outdir / "dataset_images_summary.csv", index=False)


def feature_report(features_csv: Path, outdir: Path):
    """
    dataset_features.csv is output of images_to_features.py:
    columns f0..f62 + label :contentReference[oaicite:4]{index=4}
    """
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(features_csv)
    if "label" not in df.columns:
        raise ValueError("features CSV must include 'label' column")

    counts = df["label"].astype(int).value_counts().sort_index()
    _save_bar_plot(counts, "Usable samples per class (after MediaPipe)", "Class", "Count",
                   outdir / "features_per_class.png")

    # Quick “feature magnitude” sanity plot (mean abs per feature)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    if feat_cols:
        mean_abs = df[feat_cols].abs().mean()
        plt.figure()
        plt.plot(mean_abs.values)
        plt.title("Mean |feature| (sanity check)")
        plt.xlabel("Feature index")
        plt.ylabel("Mean |value|")
        plt.tight_layout()
        plt.savefig(outdir / "feature_mean_abs.png", dpi=200)
        plt.close()

    df.to_csv(outdir / "dataset_features_summary.csv", index=False)


def training_report(metrics_json: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    data = json.loads(metrics_json.read_text(encoding="utf-8"))

    # Save a simple human-readable summary for slides
    lines = []
    for k, v in data.items():
        if isinstance(v, (int, float, str)):
            lines.append(f"{k}: {v}")
    (outdir / "training_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # Plot CV scores if present
    if "cv_scores" in data and isinstance(data["cv_scores"], list) and data["cv_scores"]:
        scores = data["cv_scores"]
        plt.figure()
        plt.plot(range(1, len(scores) + 1), scores, marker="o")
        plt.title("Cross-validation scores")
        plt.xlabel("Fold")
        plt.ylabel("Score")
        plt.tight_layout()
        plt.savefig(outdir / "cv_scores.png", dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", type=Path, help="telemetry_log.csv")
    ap.add_argument("--dataset_images", type=Path, help="dataset.csv (path,label)")
    ap.add_argument("--dataset_features", type=Path, help="dataset_features.csv (f0..f62,label)")
    ap.add_argument("--labels", type=Path, help="labels json (optional for nicer chart names)")
    ap.add_argument("--training_metrics", type=Path, help="training_metrics.json (from train_model)")
    ap.add_argument(
    "--outdir",
    type=Path,
    default=Path(__file__).resolve().parents[3] / "outputs" / "experiment_runs")
    args = ap.parse_args()

    if args.telemetry:
        telemetry_report(args.telemetry, args.outdir / "telemetry")
    if args.dataset_images:
        dataset_image_report(args.dataset_images, args.labels, args.outdir / "dataset_images")
    if args.dataset_features:
        feature_report(args.dataset_features, args.outdir / "dataset_features")
    if args.training_metrics:
        training_report(args.training_metrics, args.outdir / "training")

    print(f"Done. Outputs in: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
