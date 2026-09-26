# Top 5 traps

Each cost real time or put a wrong number in the manuscript. More in `docs/learnings/`.

1. **Which classifier produced this number?** The 149 original flights and every
   scenario trial ran the geometric rule, not the SVM: the controller then fell back
   silently when `--model`/`--labels` were missing. Only the hover-locked captures
   (short paper §V-B, Table III) flew the SVM. `--classifier` is now required, but old
   results stay as they ran. Check `manifest.json` → `classifier`, or which of
   `svm_infer_ms`/`rule_infer_ms` is populated in `perf.csv`, before quoting any
   gesture figure.
2. **Empty run directories.** `RunContext` creates the directory before the SDK
   handshake, so every failed launch leaves an empty one (these were cleared on 2026-09-25; new ones will appear). Count runs by
   `manifest.json`, as `audit_claims._runs()` does. Globbing `outputs/runs/*/`
   inflated the paper's run count for two revisions.
3. **The controller does not send what the classifier returns.**
   `DepthStabilityGate` strips FORWARD/BACK until the label repeats. Scoring
   `rule.predict()` measures the classifier, not the system, and the difference
   lands exactly on the depth classes.
4. **`%` in generated LaTeX is a comment.** A bare `59%` in a one-line generated
   paragraph deleted the rest of it from the PDF while the source looked intact.
   Emit `\%` and wrap generated prose.
5. **Scripted edits lose content.** `python - <<'PY'` and `python -c` mangle `\\`,
   `\n` and `\b` in LaTeX and regexes; a script that asserts through many
   substitutions and writes once at the end silently discards every edit when one
   misses. Write the script to a file, and write after each edit.
