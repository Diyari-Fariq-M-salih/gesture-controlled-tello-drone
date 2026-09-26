"""Apply the house table style used by the accepted VIDAR paper.

Tight column separation, a size step down, no outer padding, and a bold summary
row. Run after build_paper_tables.py; idempotent.

    python -m tello_gesture_py.src.gestures.table_style --dir paper/tables
"""

import argparse
import glob
import os
import re

PREAMBLE = ("\\centering\n"
            "\\footnotesize\n"
            "\\setlength{\\tabcolsep}{4pt}\n"
            "\\renewcommand{\\arraystretch}{1.02}\n")


def restyle(path: str) -> bool:
    with open(path, encoding="utf-8") as f:
        s = f.read()
    if "\\setlength{\\tabcolsep}" in s:
        return False

    # size + spacing block replaces the bare \centering
    s = s.replace("\\centering\n", PREAMBLE, 1)
    # a \footnotesize that was already there would now be duplicated
    s = s.replace("\\renewcommand{\\arraystretch}{1.02}\n\\footnotesize\n",
                  "\\renewcommand{\\arraystretch}{1.02}\n", 1)

    # strip outer padding from the column spec
    def fix_spec(m):
        spec = m.group(1)
        if spec.startswith("@{}"):
            return m.group(0)
        return "\\begin{tabular}{@{}" + spec + "@{}}"

    s = re.sub(r"\\begin\{tabular\}\{([^}]*)\}", fix_spec, s, count=1)

    # bold the summary row where one exists
    for key in ("Accuracy &", "Total &", "Macro $F_1$ &"):
        if key in s:
            line_start = s.index(key)
            line_end = s.index("\\\\", line_start)
            row = s[line_start:line_end]
            cells = [c.strip() for c in row.split("&")]
            bolded = " & ".join("\\textbf{%s}" % c if c else c for c in cells)
            s = s[:line_start] + bolded + s[line_end:]

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="paper/tables")
    args = ap.parse_args()
    for p in sorted(glob.glob(os.path.join(args.dir, "*.tex"))):
        print(("  restyled " if restyle(p) else "  skipped  ") + os.path.basename(p))


if __name__ == "__main__":
    main()
