# Runbook: the three captures

Follow top to bottom. Every command runs from the repo root:

```bash
cd ~/Desktop/Diyari_Fariq/projects/gesture-controlled-tello-drone
```

Sessions 1 and 2 need the drone and take about 40 minutes together. Session 3
needs daylight and a webcam only, about 25 minutes, and can be another day.

---

## Before anything

**Activate the environment.** Every command below assumes it.

```bash
source .venv/Scripts/activate      # Git Bash
# .venv\Scripts\activate.bat       # cmd
```

**Check the model loads clean.**

```bash
python -c "from tello_gesture_py.src.model_classifier import TrainedClassifier as T, _resolve_model_path as R; print(R('model.joblib')); T('model.joblib','labels_example.json'); print('OK')"
```

Must print a path ending `models\production\model.joblib` and then `OK`, with
**no** `InconsistentVersionWarning`. If you see that warning, the model was
pickled by a different scikit-learn than the one running and its predictions
cannot be trusted. Install the scikit-learn version pinned in
`tello_gesture_py/requirements-lock.txt`. **Do not retrain:** the model is
frozen and its SHA-256 is preregistered (`docs/PREREGISTRATION_svm_inflight.md`);
a retrained model is a different classifier from the one the paper measures.

---

# Session 1 — Cued gesture capture

**~20 min · drone · PROPELLERS OFF**

The measurement the paper is missing. Also settles whether the SVM and the rule
path agree, which matters because the SVM has never actually flown.

## Setup

1. **Take the propellers off.** All four. The scripts send only `command`,
   `streamon` and `streamoff` — there is no `takeoff` or `rc` anywhere in them —
   but you will be half a metre from a powered drone in SDK mode for 20 minutes,
   and anything else that reaches port 8889 (a stale `main.py`, the Tello phone
   app reconnecting) can launch it into your face.
2. Kill any other Python talking to the drone. One process owns port 9013.
3. Drone on a table at roughly chest height, camera facing where you will stand.
4. Power on. Wait for the status LED to stop flashing amber.
5. Join the `TELLO-XXXXXX` Wi-Fi. Pin it so Windows does not wander off to a
   network with internet:

   ```bash
   netsh wlan set profileparameter name="eduroam" connectionmode=manual
   netsh wlan set profileparameter name="PlanetCampus" connectionmode=manual
   ```

6. Confirm the link and battery:

   ```bash
   python -m tello_gesture_py.scripts.tello_check
   ```

   Charge above ~30% is plenty; nothing here flies. If it reports no route,
   you are on the wrong Wi-Fi.

7. Stand where you would stand to fly it — **1.5–2 m back**, so apparent hand
   size matches flight. Do not improve the lighting. Whatever the room is, is
   the measurement.

## Run

```bash
python -m tello_gesture_py.scripts.operational_gesture_eval --model model.joblib --labels labels_example.json --run-id capture_gesture-svm --note "ground, props off, living room, afternoon daylight"
```

Change `--note` to match the actual room and light. It goes in the manifest and
is what lets anyone reading the logs know what condition produced them.

`--model` is **not optional**. Without it the SVM columns come back empty and
the capture cannot answer the question the paper asks.

## What happens

42 holds, 6 rounds of 7 classes, class order randomised within each round so
fatigue or light drift cannot align with one class.

Each hold:

| | |
|---|---|
| Window turns **amber**, shows `SETTLE  <CLASS>` | change into the gesture — 5 s, discarded |
| Window turns **green**, shows `HOLD  <CLASS>` | hold it still — 6 s, recorded |

Hold steady and keep your hand in frame. The HUD shows the live rule and SVM
predictions; **ignore them**. Do not adjust your hand to make the prediction
agree, and do not redo a hold because it looked wrong. That is the measurement.

Do let your hand drift naturally, turn slightly, sit at the edge of frame
sometimes. Those frames are the operational condition.

## Keys

| key | effect |
|---|---|
| `SPACE` | pause — **stops the hold clock**, so rest your arm freely. `SPACE` again to resume |
| `s` | skip this hold (use if you fumbled the start) |
| `q` | abort, keeping everything recorded so far |

If you abort, or the stream dies:

```bash
python -m tello_gesture_py.scripts.operational_gesture_eval --model model.joblib --labels labels_example.json --resume outputs/runs/<the-directory-it-printed>
```

Holds already recorded are skipped; it picks up where you stopped.

## Done when

It prints `N hold-window frames -> .../op_gesture.csv`. Expect roughly
5000–6000 frames. Under ~2500 means the stream was poor — check the reopen count
in the manifest and consider rerunning at a quieter time.

---

# Session 2 — Authorization envelope

**~15 min · drone · PROPELLERS OFF · same sitting as Session 1**

Fixes Figure 2 resting on bins of n=2 and n=3, and resolves the unexplained
32.0% ground versus 19.3% flight inversion.

## Setup

1. Props still off. Drone still powered and streaming.
2. Move it to **eye height**, facing down a clear run of floor.
3. **Tape eight marks**: 0.5, 0.8, 1.0, 1.5, 1.75, 2.0, 2.5, 3.0 m. Not evenly
   spaced on purpose - see below. Measure once, then trust the marks.
4. Clear the background behind you of other people.

## Run

```bash
python -m tello_gesture_py.scripts.stream_capture --run-id capture_envelope --note "ground, props off, 0.5-3.0 m marked, living room" --walkup-s 3
```

`--walkup-s` is the gap between pressing a distance key and recording starting,
so you can walk to the mark. Frames during the walk are **discarded, not
binned** — otherwise the first seconds of every segment would be you in transit,
filed under a distance you were not standing at. Raise it if your marks are far
apart or you want to take it slowly.

## Procedure

1. Stand on the **1.5 m** mark. Press `p`. Hold still, face the drone, neutral
   expression. The HUD counts enrolment crops to 20, then shows `enrolled=Y`.
   This template is used for the whole sweep — do not re-enrol partway.
2. For each mark, nearest to furthest:
   - press its key: `1`=0.5, `2`=0.8, `3`=1.0, `4`=1.5, `5`=1.75, `6`=2.0, `7`=2.5, `8`=3.0 m
   - **walk to the mark.** The overlay shows `WALK TO 2.0 m ... 2.4 s` in amber
     and records nothing yet
   - when it turns green and reads `RECORDING 2.0 m`, hold 20–30 s facing the
     drone with **small natural head movement** — look slightly left, slightly
     right, slightly down. Not a statue, not a dance.
   - press `SPACE`
   - walk back to the keyboard for the next mark
3. **Do the whole sweep twice.**

## Reading the HUD

```
RECORDING 2.0 m   frames=418
bbox_h=312   bin=280-350 147/200   score=0.681   auth=1
short: 0-150:12, 150-180:88, 480-620:41
```

Amber overlay means walking or stopped, green means recording.

`short:` lists every bin still under the 200-frame target. Use it — if
`480-620` is short, spend longer on the 0.5 m and 1.0 m marks; if `0-150` is
short, spend longer at 3.5–4.0 m. Steering by this is the difference between a
useful sweep and discovering afterwards that a bin is empty.

## The near marks matter most

At 0.5 m and 1.0 m authorization will mostly **fail**. Capture them properly
anyway, full 20–30 s, both passes. The hypothesis those frames test is that the
ground condition contains a near-field regime flight rarely visits, where the
padded crop clips the frame edge — which would explain why the ground rate looks
worse than the flight rate. Those failures are the explanation, not noise.

## Done when

The exit table shows no `SHORT` rows:

```
bbox bin       scored  target
0-150             214     200
150-180           236     200
...
```

Any bin still short, restart the script (it writes a new run; the analysis pools
every run) and top up just that distance.

---

# Session 3 — Session D dataset

**~25 min · webcam only · needs DAYLIGHT**

The cell that breaks the confound. A is daylight + uniform, B and C are
artificial + non-uniform. D is **daylight + non-uniform**, which is the only
way to separate illumination from background.

## Setup

1. Daylight. Curtains open, no lamps on. Late morning or afternoon.
2. Stand in front of a **domestic background with visible structure** — a
   bookshelf, furniture, a patterned wall. Not a blank wall.
3. Same webcam and roughly the same distance you used for B and C.

## Run

```bash
python -m tello_gesture_py.src.gestures.auto_collect_dataset --session D --illumination daylight --background non-uniform --background-description "living room, bookshelf and sofa" --distance-m 1.5 --out_dir data/raw/sessionD --out_csv data/processed/sessionD.csv
```

Set `--distance-m` to what you actually stand at, measured. It was never
recorded for A, B or C, which is exactly why §V-C cannot attribute the
FORWARD/BACK transfer failure — don't repeat that.

`--session` refuses to start without `--illumination` and `--background`,
checked up front rather than after 2800 images.

## Procedure

Per class: 10 s countdown to get into position, then 400 frames at 200 ms
intervals (~80 s), then 10 s before the next class. Seven classes, ~20 minutes.

Hold each gesture as you did for A, B and C. `q` quits and keeps what it has.

## Then extract features

```bash
python -m tello_gesture_py.src.gestures.images_to_features --dataset data/processed/sessionD.csv --out data/processed/sessionD_features.csv
```

Expect ~2600–2800 kept of 2800. A large drop means MediaPipe struggled with the
background, which is itself a result — report it, don't recapture.

## D may score worse than B and C

That is the design. It is the discriminating cell, not a better one. Do not
adjust lighting, distance or background to improve its numbers.

---

# After all three

Run the whole pipeline:

```bash
python -m tello_gesture_py.src.gestures.per_session_eval
python -m tello_gesture_py.src.gestures.loso_eval
python -m tello_gesture_py.scripts.analyse_operational_gesture
python -m tello_gesture_py.src.gestures.envelope_eval
python -m tello_gesture_py.src.gestures.build_paper_tables --outdir outputs/paper
python -m tello_gesture_py.src.gestures.build_paper_figures --outdir paper/figures
python -m tello_gesture_py.src.gestures.audit_claims
```

Nothing needs editing for session D — the evaluations discover it and the
figure widens to four columns on its own.

`loso_eval` will now print a populated pair summary instead of the current
`not identifiable`, and its verdict line names whether illumination or
background dominates, and by how much. That is the numerical answer §V-C
currently gestures at.

`audit_claims` exits non-zero if any manuscript number stopped reproducing.
Run it before posting anything.

---

# If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `No route to 192.168.10.1` | Windows left the drone AP | rejoin `TELLO-XXXXXX`, pin other profiles to manual |
| `Failed to enter SDK mode` | another process holds port 9013 | close other Python, wait 5 s, retry |
| `No frames decoded after 12 s` | firewall blocking UDP 11111 | allow Python through Windows Firewall on private networks |
| Stream stutters, low frame count | RF congestion | move away from access points; the domestic site gave 19.7–22.3 fps, the university building 8.4 |
| `InconsistentVersionWarning` on load | model pickled by another sklearn | retrain (command at the top) |
| Window frozen, no response | OpenCV window lost focus | click the video window; keys only register while it has focus |
