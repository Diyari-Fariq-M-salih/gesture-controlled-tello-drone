# Commands

Repo root, `.venv` active (CPython 3.12). One line each: cmd.exe does not
continue a line on a trailing backslash.

## Check the paper still holds
```bash
python -m tello_gesture_py.src.gestures.audit_claims
```
Must exit 0. V-H (offline EER 0.32%) prints as EXTERNAL: no script here derives it.

## Tests (no camera, no drone)
```bash
python -m tello_gesture_py.tests.test_classifier_selection
python -m tello_gesture_py.tests.test_hand_association
python -m tello_gesture_py.tests.test_session_recorder
```
pytest is not in the requirements; each test file also runs under it if installed.

## Fly
```bash
python -m tello_gesture_py.scripts.tello_check
python -m tello_gesture_py.src.main --classifier svm --run-id <tag>
python -m tello_gesture_py.src.main --classifier rule --run-id <tag>
```
`--classifier` is required. `svm` defaults to the frozen `models/production/model.joblib`
and `labels_example.json`; pass `--model`/`--labels` only to override. Face following
is not a flag: it is the arbitration mode below gesture. Useful: `--no-fly` (RC held
at zero), `--no-llm`, `--note "<conditions>"`, `--record` (window to `session.mp4`,
`--record-raw` adds the clean frames; videos are git-ignored).

Cued, hover-locked gesture capture in flight (the paper's §V-B protocol):
`--cue-rounds 6 --no-actuate`. Never run cues without `--no-actuate` indoors: a 6 s
hold travels about 3.9 m.

Webcam only, no drone: `python -m tello_gesture_py.scripts.webcam_demo --run-id <tag>`

Hand-face association (opt-in; this operator gestures left-handed): add
`--associate-hands --target-side left` to either command; `--debug-overlay --live-pose`
draws the landmarks and tracks the body every frame (keys `v`, `b`). See
`docs/learnings/hand-association.md`.

## Captures (propellers off, drone streaming)
```bash
python -m tello_gesture_py.scripts.operational_gesture_eval --model model.joblib --labels labels_example.json --run-id capture_gesture-svm --note "ground, props off"
python -m tello_gesture_py.scripts.operational_gesture_eval --vocabulary pointing --run-id capture_gesture-rule --note "ground, props off"
python -m tello_gesture_py.scripts.stream_capture --run-id capture_envelope --walkup-s 3
python -m tello_gesture_py.scripts.operational_gesture_eval --resume outputs/runs/<dir>
```
Envelope marks: 0.5 0.8 1.0 1.5 1.75 2.0 2.5 3.0 m. Procedures: `docs/RUNBOOK.md`, `docs/KEYS.md`.

Camera-to-host delay (drone facing this screen, props off; `--selftest 120` checks the
method without a drone):
```bash
python -m tello_gesture_py.scripts.flash_latency
```
Command-to-motion (flying, hovering; `t` takes off, `g` starts, `e` emergency;
`--selftest 150 600` checks the method without a drone):
```bash
python -m tello_gesture_py.scripts.yaw_latency --video-delay-ms 604
```

## Analyse
```bash
python -m tello_gesture_py.scripts.analyse_operational_gesture --run outputs/runs/<dir>
python -m tello_gesture_py.scripts.analyse_operational_gesture --exclude LEFT:0-2 --exclude RIGHT:0-2
python -m tello_gesture_py.src.gestures.envelope_eval
```

## Regenerate tables and figures
```bash
python -m tello_gesture_py.src.gestures.per_session_eval
python -m tello_gesture_py.src.gestures.loso_eval
python -m tello_gesture_py.src.gestures.build_paper_tables --outdir outputs/paper
python -m tello_gesture_py.src.gestures.build_paper_figures --outdir paper/figures
python -m tello_gesture_py.src.gestures.fig_envelope_distance
python -m tello_gesture_py.src.gestures.build_design_table --short
python -m tello_gesture_py.src.gestures.build_design_figures
```
`fig_envelope_distance` is the only source of Fig. 3; `build_paper_figures` skips it on purpose.
Fig. 4 (`fig_offline_eer.png`) is committed; rebuilding it needs page renders of the
project report, which is not in the repository:
`python -m tello_gesture_py.src.gestures.build_offline_eer_figure --src <renders dir>`.

## Build the papers
```bash
cd paper && pdflatex -interaction=nonstopmode short.tex && bibtex short && pdflatex -interaction=nonstopmode short.tex && pdflatex -interaction=nonstopmode short.tex
```
`short.tex` is the arXiv paper; swap in `main.tex` for the 13-page version. The arXiv
source as submitted is `paper/arxiv-v1/` (frozen), also tag `arxiv-v1`.

## If the model load warns `InconsistentVersionWarning`
Fix the scikit-learn version in `.venv`. Never retrain: the model is frozen.
