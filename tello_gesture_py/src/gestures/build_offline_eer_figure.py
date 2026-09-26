"""Stack the three offline face-verification panels into one 3x3 figure.

The panels are the co-author's offline benchmark, rendered from the M2 report
rather than recomputed here: no script in this repository derives them, and the
audit records the 0.32% equal error rate as EXTERNAL for that reason. This
script only crops and stacks, so it cannot change a number.

Each row is one negative set, growing left to right through the same three
views: score distribution, ROC, and the FAR/FRR trade-off against threshold.

    python -m tello_gesture_py.src.gestures.build_offline_eer_figure
"""

import argparse
import os

import cv2
import numpy as np

# (page, first row, last row) of each panel block in the rendered report.
BLOCKS = [("pg-08.png", 1421, 1757, "(a)  100 impostor images"),
          ("pg-09.png", 208, 541, "(b)  $N=4{,}678$"),
          ("pg-09.png", 663, 995, "(c)  $N=13{,}410$")]
GAP = 14        # px between rows
PAD = 8


def crop(src_dir, name, r0, r1):
    im = cv2.imread(os.path.join(src_dir, name))
    if im is None:
        raise SystemExit("missing render: %s" % name)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[r0:r1 + 1]
    cols = np.where((g < 200).sum(axis=0) > 0)[0]
    c0 = max(0, cols.min() - PAD)
    c1 = min(im.shape[1], cols.max() + PAD + 1)
    return im[max(0, r0 - PAD):r1 + PAD + 1, c0:c1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="directory of pdftoppm renders of the report pages")
    ap.add_argument("--out", default="paper/figures/fig_offline_eer.png")
    a = ap.parse_args()

    rows = [crop(a.src, n, r0, r1) for n, r0, r1, _ in BLOCKS]
    w = max(r.shape[1] for r in rows)
    canvas = []
    for i, r in enumerate(rows):
        if r.shape[1] != w:                      # centre narrower rows
            pad = w - r.shape[1]
            r = cv2.copyMakeBorder(r, 0, 0, pad // 2, pad - pad // 2,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
        canvas.append(r)
        if i != len(rows) - 1:
            canvas.append(np.full((GAP, w, 3), 255, np.uint8))
    out = np.vstack(canvas)
    cv2.imwrite(a.out, out)
    print("wrote %s  %dx%d" % (a.out, out.shape[1], out.shape[0]))


if __name__ == "__main__":
    main()
