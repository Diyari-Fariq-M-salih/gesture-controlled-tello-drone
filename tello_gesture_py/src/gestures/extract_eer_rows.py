"""Extract the three offline-EER panel rows from the report, separately.

Two forms per row, because they are useful for different things:

  * a cropped **vector** PDF, lifted straight out of the report with no
    rasterisation, so it can be rescaled or re-laid-out without loss;
  * a 600 dpi PNG, for editing in a raster tool.

Crop boxes are found by ink profile rather than typed in, so a re-render at a
different resolution still lands on the panels.

    python -m tello_gesture_py.src.gestures.extract_eer_rows --report <project report.pdf> --outdir paper/figures/eer_rows

The project report is not in this repository; the extracted rows are.
"""

import argparse
import glob
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

PROBE_DPI = 200
PNG_DPI = 600

# (report page, label) of each panel row, in the order they appear.
ROWS = [(8, "row1_n176"), (9, "row2_n4678"), (9, "row3_n13410")]
BAND_MIN_H = 250          # a panel row is tall; captions and body text are not
GAP = 12                  # blank rows that separate one band from the next


def bands(gray):
    ink = (gray < 200).sum(axis=1)
    rows = np.where(ink > 0)[0]
    out, start, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - prev > GAP:
            out.append((start, prev))
            start = r
        prev = r
    out.append((start, prev))
    return [(a, b) for a, b in out if b - a >= BAND_MIN_H]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--outdir", default="paper/figures/eer_rows")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    tmp = tempfile.mkdtemp()
    # LaTeX dislikes spaces in graphics paths, so work on a copy.
    src = os.path.join(tmp, "report.pdf")
    shutil.copy(a.report, src)

    pages = sorted({p for p, _ in ROWS})
    for p in pages:
        subprocess.run(["pdftoppm", "-f", str(p), "-l", str(p),
                        "-r", str(PROBE_DPI), "-png", src,
                        os.path.join(tmp, "p%d" % p)], check=True)

    found = {}
    for p in pages:
        f = glob.glob(os.path.join(tmp, "p%d-*.png" % p))[0]
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        found[p] = (g.shape, bands(g), f)

    # Assign bands to rows in page order.
    per_page = {}
    for p, label in ROWS:
        per_page.setdefault(p, []).append(label)

    k = 72.0 / PROBE_DPI
    for p, labels in per_page.items():
        (H, W), bs, png = found[p]
        if len(bs) < len(labels):
            raise SystemExit("page %d: found %d panel bands, expected %d"
                             % (p, len(bs), len(labels)))
        use = bs[-len(labels):] if p == 8 else bs[:len(labels)]
        big = cv2.imread(png)
        for (r0, r1), label in zip(use, labels):
            seg = cv2.cvtColor(big[r0:r1 + 1], cv2.COLOR_BGR2GRAY)
            cols = np.where((seg < 200).sum(axis=0) > 0)[0]
            c0, c1 = max(0, cols.min() - 8), min(W, cols.max() + 9)
            y0, y1 = max(0, r0 - 8), min(H, r1 + 9)

            # --- vector crop -------------------------------------------
            tex = os.path.join(tmp, label + ".tex")
            with open(tex, "w", encoding="utf-8") as fh:
                fh.write(
                    "\\documentclass{article}\n"
                    "\\usepackage[margin=0pt,paperwidth=%.2fpt,paperheight=%.2fpt]{geometry}\n"
                    "\\usepackage{graphicx}\\pagestyle{empty}\n"
                    "\\begin{document}\\noindent\n"
                    "\\includegraphics[page=%d,trim=%.2f %.2f %.2f %.2f,clip]{report}\n"
                    "\\end{document}\n"
                    % ((c1 - c0) * k, (y1 - y0) * k, p,
                       c0 * k, (H - y1) * k, (W - c1) * k, y0 * k))
            subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "-output-directory", tmp, tex],
                           cwd=tmp, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            made = os.path.join(tmp, label + ".pdf")
            if os.path.exists(made):
                shutil.copy(made, os.path.join(a.outdir, label + ".pdf"))

            # --- high-resolution raster --------------------------------
            s = PNG_DPI / float(PROBE_DPI)
            subprocess.run(["pdftoppm", "-f", str(p), "-l", str(p),
                            "-r", str(PNG_DPI), "-png",
                            "-x", str(int(c0 * s)), "-y", str(int(y0 * s)),
                            "-W", str(int((c1 - c0) * s)),
                            "-H", str(int((y1 - y0) * s)),
                            src, os.path.join(a.outdir, label)], check=True)
            print("  %-16s vector + %d dpi png" % (label, PNG_DPI))

    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", a.outdir)


if __name__ == "__main__":
    main()
