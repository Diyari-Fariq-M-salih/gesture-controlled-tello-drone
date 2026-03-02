import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2


def load_labels(labels_json_path: str):
    with open(labels_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Supports either:
    # 1) {"labels": [{"id":0,"name":"CENTER"}, ...]}
    # 2) {"0":"CENTER","1":"LEFT", ...}
    if isinstance(data, dict) and "labels" in data:
        labels = [(int(x["id"]), str(x["name"])) for x in data["labels"]]
    elif isinstance(data, dict):
        labels = [(int(k), str(v)) for k, v in data.items()]
    else:
        raise ValueError("Unsupported labels JSON format.")

    labels.sort(key=lambda x: x[0])
    return labels


def put_overlay(frame, lines, y0=30, dy=28):
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, y0 + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def countdown(cap, seconds, label_name, collected, total, window="DATASET"):
    end = time.time() + seconds
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        remaining = int(max(0, end - time.time()))
        put_overlay(
            frame,
            [
                f"CLASS: {label_name}",
                f"Collected: {collected}/{total}",
                f"Get ready... {remaining}s",
                "Press 'q' to quit",
            ],
        )
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if time.time() >= end:
            return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labels",
        default="data/labels/labels_example.json",
        help="Path to labels JSON"
    )

    ap.add_argument("--out_dir", default="dataset_images", help="Where to save images")
    ap.add_argument("--out_csv", default="dataset.csv", help="CSV output file")
    ap.add_argument("--cam", type=int, default=0, help="Webcam index (0 is default)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)

    # Your controls:
    ap.add_argument("--images_per_class", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=30)
    ap.add_argument("--ready_delay", type=float, default=5.0)
    ap.add_argument("--between_batch_delay", type=float, default=5.0)
    ap.add_argument("--capture_interval_ms", type=int, default=120, help="Time between captures")

    args = ap.parse_args()

    # Resolve project root (…/tello_gesture_py/src/gestures -> …/project-root)
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    args.images_per_class = 400
    args.batch_size = 400
    args.ready_delay = 10
    args.between_batch_delay = 10
    args.capture_interval_ms = 200

    args.labels = "data/labels/labels_example.json"
    args.out_dir = "dataset_images"
    args.out_csv = "dataset.csv"

    labels = load_labels(args.labels)
    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.cam}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    window = "DATASET"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print("Labels:")
    for lid, name in labels:
        print(f"  {lid}: {name}")
    print("\nPress 'q' anytime to quit.\n")

    try:
        for label_id, label_name in labels:
            class_dir = os.path.join(args.out_dir, f"{label_id:02d}_{label_name}")
            os.makedirs(class_dir, exist_ok=True)

            collected = 0
            total = args.images_per_class

            while collected < total:
                # 5s get-ready
                ok = countdown(
                    cap,
                    args.ready_delay,
                    label_name,
                    collected,
                    total,
                    window=window,
                )
                if not ok:
                    raise KeyboardInterrupt

                # capture a batch
                batch_target = min(args.batch_size, total - collected)
                for _ in range(batch_target):
                    ok, frame = cap.read()
                    if not ok:
                        continue

                    fname = f"{run_id}_L{label_id:02d}_{collected:06d}.jpg"
                    fpath = os.path.join(class_dir, fname)

                    cv2.imwrite(fpath, frame)
                    rows.append((fpath, label_id))

                    collected += 1

                    put_overlay(
                        frame,
                        [
                            f"CLASS: {label_name}",
                            f"Saved: {collected}/{total}",
                            f"Batch size: {batch_target}",
                            "Press 'q' to quit",
                        ],
                    )
                    cv2.imshow(window, frame)
                    key = cv2.waitKey(args.capture_interval_ms) & 0xFF
                    if key == ord("q"):
                        raise KeyboardInterrupt

                if collected < total:
                    # 5s between batches
                    end = time.time() + args.between_batch_delay
                    while time.time() < end:
                        ok, frame = cap.read()
                        if not ok:
                            continue
                        remaining = int(end - time.time())
                        put_overlay(
                            frame,
                            [
                                f"CLASS: {label_name}",
                                f"Collected: {collected}/{total}",
                                f"Next batch in {remaining}s",
                                "Press 'q' to quit",
                            ],
                        )
                        cv2.imshow(window, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            raise KeyboardInterrupt

        # write CSV
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            w.writerows(rows)

        print(f"\nDone. Saved images to: {args.out_dir}")
        print(f"Wrote CSV to: {args.out_csv}")

    except KeyboardInterrupt:
        # still write what we have
        if rows:
            with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["path", "label"])
                w.writerows(rows)
            print(f"\nStopped early. Wrote partial CSV to: {args.out_csv}")
        else:
            print("\nStopped early. No data written.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
