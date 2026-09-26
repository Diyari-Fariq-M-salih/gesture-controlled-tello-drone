# Capture protocol: operational gesture accuracy, session D, and the envelope

Three captures, none requiring flight. Run each from the repo root with `.venv`
active, and joined to the drone's access point for captures 1 and 3.

No gating, arbitration, threshold or embedding parameter changes for any of
these. The harnesses read the deployed classes and write into the existing
`outputs/runs/<stamp>_<tag>/` layout with the usual manifest (config, git SHA,
library versions, host).

---

## Read this before capture 1

Every one of the 43 logged runs in this repository carries
`classifier: rule_based`, and `--model` was never passed on any command line.
**The RBF-SVM that Section III-B describes and Section V-C evaluates has never
been in the flight loop.** The compound labels in the decision logs
(`DOWN-FORWARD`, `UP-FORWARD`) are `RuleBasedGesture` outputs; the SVM emits
single class names from `labels.json`.

So the harness records **both** classifiers on identical frames:

- the **SVM**, because it is what the paper characterises and what the
  offline/operational comparison has to be about;
- the **rule path**, because it is what actually produced every flight result
  reported, including Section V-G's depth contamination.

Both are deployed classes that `Controller.__init__` selects between; neither is
reimplemented and neither commands the aircraft here. Pass `--model` or the SVM
columns come back empty and the capture cannot answer the paper's question.

---

## 1. Operational gesture accuracy — cued, drone stream, props off

```bash
python -m tello_gesture_py.scripts.operational_gesture_eval --model model.joblib --labels labels_example.json --run-id capture_gesture-svm --note "ground, props off, domestic room, daylight"
```

**Setup.** Propellers off. Drone on a table at roughly chest height, powered on,
streaming. Stand where you would stand to fly it, about 1.5–2 m back, so
apparent hand size matches flight. Do not stage the lighting.

**Protocol.** The script drives it. For each of 42 holds it names a class, gives
you `SETTLE_S` (1.5 s) to form and hold the gesture, then records `HOLD_S`
(3.0 s) of frames labelled with that class. Settle-window frames are discarded.
Class order is randomised within each of 6 rounds, so operator fatigue or
lighting drift cannot align with one class.

Defaults give ~42 holds × ~3 s ≈ 2.5 min of recorded holding, ~20 min wall clock.
Tune with `--settle-s`, `--hold-s`, `--rounds`.

**Keys.** `SPACE` pauses and stops the hold clock so you can rest your arm.
`s` skips the current hold. `q` aborts, keeping everything captured so far —
resume with `--resume outputs/runs/<dir>`, which skips holds already recorded.

**Per frame it logs** sequence number, timestamp, frame age, cued label,
hand-detected flag, bbox area, the depth cue `delta_s`, the rule label and
confidence, and the SVM label, probability and top-two margin.

`delta_s` is read out of the deployed rule object rather than recomputed:
`RuleBasedGesture.predict()` stores the smoothed scale in `_last_scale` and
forms its cue as `(s_t - s_{t-1}) / s_{t-1}`, so sampling that attribute either
side of the call recovers the value the classifier acted on without touching it.

**Do not re-do a hold because the prediction looked wrong.** That is the
measurement.

**Analysis.**

```bash
python -m tello_gesture_py.scripts.analyse_operational_gesture
```

Produces, per classifier: accuracy over all cued frames and over detected-hand
frames separately, detection rate, per-class F1, and a row-normalised confusion
matrix with an eighth "no detection" column, drawn in the Figure 1 idiom so it
can sit beside the within-session and held-out matrices. Writes
`outputs/metrics/operational_gesture.json`, `paper/figures/fig_operational.pdf`
and `paper/tables/tab_operational.tex`.

It also regenerates Section V-G from this cued sample, splitting two outcomes
the opportunistic count conflated: **compound contamination** (a lateral label
carrying a depth component, e.g. `DOWN-FORWARD` — an uncommanded depth velocity
alongside a correct axis, which is what V-G measures) from **outright depth
misclassification** (a bare `FORWARD` on a cued `LEFT`, already counted in the
confusion matrix). Pooling them lets classifier error inflate the contamination
rate.

---

## 2. Session D — webcam, daylight, non-uniform background

The cell that breaks the confound. A is daylight + uniform; B and C are
artificial + non-uniform; D is daylight + non-uniform.

```bash
python -m tello_gesture_py.src.gestures.auto_collect_dataset --session D --illumination daylight --background non-uniform --background-description "living room, bookshelf and furniture" --distance-m 1.5 --out_dir data/raw/sessionD --out_csv data/processed/sessionD.csv

python -m tello_gesture_py.src.gestures.images_to_features --dataset data/processed/sessionD.csv --out data/processed/sessionD_features.csv
```

`--session` requires `--illumination` and `--background`, checked before capture
rather than after 2800 images. On completion the session is registered in
`data/sessions.json`, which is what lets the transfer analysis group pairs by
condition. Sessions A, B and C are already backfilled there from the capture
timestamps, the derived photometry and the manuscript's descriptions;
`nominal_distance_m` is `null` for all three because it was never recorded,
which is itself why Section V-C cannot attribute the FORWARD/BACK transfer
failure.

Capture parameters default to what B and C used (400/class, 200 ms interval,
10 s ready), so the protocol is held constant.

**D may score worse than B and C. That is the design** — it is the
discriminating cell, not a better one. Do not adjust lighting, distance or
background to improve it.

---

## 3. Authorization envelope and on-ground FRR — drone stream, props off

```bash
python -m tello_gesture_py.scripts.stream_capture --run-id capture_envelope --note "ground, props off, 0.5-4.0 m marked"
```

**Setup.** Propellers off. Drone fixed at eye height, streaming, pointing down a
clear run of floor. Mark the floor at 0.5 m intervals from 0.5 m to 4.0 m with
tape — measure once, then trust the tape.

**Procedure.** Stand at 1.5 m, press `p`, hold still until the HUD shows
enrolled. Then for each mark: stand on it, press the matching key (`1` = 0.5 m …
`8` = 4.0 m), hold 20–30 s facing the drone with small natural head movement,
press `SPACE`, step to the next. Sweep twice.

The HUD shows which bin the current bbox height falls in, its fill against the
200-frame target, and a `short:` line naming every bin still under target, so
you can steer the sweep toward the empty bins instead of finding out afterwards.
The same counts print at exit and go into the manifest.

**On the ground-versus-flight inversion.** The hypothesis worth testing while
you are set up: the ground condition *contains* a near-field regime flight
rarely visits, where the padded crop clips the frame edge. The 0.5 m and 1.0 m
marks decide it. Capture them properly even though authorization will mostly
fail there — those frames are the explanation, not noise.

**Analysis.**

```bash
python -m tello_gesture_py.src.gestures.envelope_eval
```

Authorization by bbox height and by declared distance, with Wilson intervals and
per-bin counts, and the FRR split by whether a face was detected at all — a
frame with no detection carries no authorization decision and is not a rejection
by the embedding.

---

## After capturing

```bash
python -m tello_gesture_py.src.gestures.per_session_eval
python -m tello_gesture_py.src.gestures.loso_eval
python -m tello_gesture_py.scripts.analyse_operational_gesture
python -m tello_gesture_py.src.gestures.envelope_eval
python -m tello_gesture_py.src.gestures.build_paper_tables --outdir outputs/paper
python -m tello_gesture_py.src.gestures.build_paper_figures --outdir paper/figures
python -m tello_gesture_py.src.gestures.audit_claims
```

`loso_eval` picks up session D with no edit, runs four-fold LOSO, keeps the
published A–C result alongside it, prints the 4×4 transfer matrix, and groups
every ordered pair by what it holds constant:

| group | shares illumination | shares background |
|---|---|---|
| `both_shared` | yes | yes |
| `illumination_only` | yes | no |
| `background_only` | no | yes |
| `neither` | no | no |

With A, B and C alone the two middle cells are **empty** — no pair shares one
factor without the other — and the script says so rather than guessing. Session
D fills them, and the verdict line then names which factor dominates and by how
much. That is the numerical answer to the illumination-versus-background
question.

`audit_claims` rederives every number in the manuscript from the logs and exits
non-zero if any fails. Run it before posting.
