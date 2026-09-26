# Keystroke sequence — all three captures

Literal order of operations. Keys only register while the **video window has
focus**, so click it once after each script starts.

Every command below is one line. Paste it whole — do not split it. (In cmd.exe a
trailing backslash is not a line continuation, it is an error; the continuation
character there is `^`.)

---

## TEST 1 — Gestures (drone, props off, ~8 min)

### Before

```
[ ] four propellers OFF
[ ] no other Python running (one process owns port 9013)
[ ] drone on table, chest height, camera facing you
[ ] drone powered on, LED done flashing amber
[ ] joined TELLO-XXXXXX wifi
[ ] you standing 1.5-2 m back
```

```bash
python -m tello_gesture_py.scripts.tello_check
```

Expect a battery reading. If it says no route, you are on the wrong Wi-Fi.

### Start

```bash
python -m tello_gesture_py.scripts.operational_gesture_eval --model model.joblib --labels labels_example.json --run-id capture_gesture-svm --note "ground, props off, living room, afternoon daylight"
```

Click the video window.

### Then: no keys at all

**You press nothing for the next ~8 minutes.** The script drives all 42 holds.
Your only job is your hand.

Each hold, twice a minute:

```
  AMBER   SETTLE  LEFT        <- 5 s to change into LEFT, nothing recorded
  GREEN   HOLD    LEFT  4.2s  <- 6 s held steady, in frame, recorded
  AMBER   SETTLE  UP          <- 5 s to change to UP
```

Order is randomised. Watch the class name, not the predictions.

### Keys only if needed

| key | when |
|---|---|
| `SPACE` | arm tired. Pauses and **stops the clock**. `SPACE` again to carry on |
| `s` | you fumbled the start of this hold — skips it, moves to the next |
| `q` | stop for good. Everything so far is kept |

### If you pressed `q` or the stream died

```bash
python -m tello_gesture_py.scripts.operational_gesture_eval --model model.joblib --labels labels_example.json --resume outputs/runs/<directory it printed>
```

Completed holds are skipped automatically.

### Done

Prints `N hold-window frames`. Expect ~5000-6000 over 42 holds.

### Rules

- **Do not** change your hand to make the on-screen prediction agree.
- **Do not** redo a hold that looked wrong.
- **Do** let your hand drift, turn, sit near the frame edge sometimes.

---

## TEST 2 — Distance sweep (drone, props off, ~15 min)

### Before

```
[ ] props still off, drone still powered
[ ] drone moved to EYE height, facing down a clear floor run
[ ] tape marks at 0.5 0.8 1.0 1.5 1.75 2.0 2.5 3.0 m  (measure once)
[ ] nobody else in frame behind you
```

### Start

```bash
python -m tello_gesture_py.scripts.stream_capture --run-id capture_envelope --note "ground, props off, 0.5-3.0 m marked, living room" --walkup-s 3
```

Click the video window.

### Enrol — once, at the start

```
stand on the 1.5 m mark
press  p
hold still, face the drone, neutral expression
wait for HUD to read  enrolled=Y        (counts crops to 20)
```

**Do not press `p` again for the rest of the session.** One template for the
whole sweep, or the distances are not comparable.

### The sweep — repeat per mark

```
press  1        <- 0.5 m   (2=0.8  3=1.0  4=1.5  5=1.75  6=2.0  7=2.5  8=3.0)
walk to the mark            <- AMBER: "WALK TO 0.5 m ... 2.4s", records nothing
hold 20-30 s                <- GREEN: "RECORDING 0.5 m"
   face the drone
   small natural head movement: slightly left, slightly right, slightly down
   not a statue, not a dance
press  SPACE    <- stops recording
walk back to keyboard
```

Then the next mark. Nearest to furthest:

```
1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8
```

### Then do the whole run again

```
1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8
```

### Steering by the HUD

Bottom line names every bin still under 200 frames:

```
short: 0-150:12, 150-180:88, 480-620:41
```

- `480-620` short → more time at **0.8 m**
- `0-150` short → more time at **3.0 m**

Go back and top up whichever is short. Order does not matter for topping up.

### Do not skip the near marks

At 0.5 m authorization will mostly read `auth=0`. **Capture them
anyway, full 20–30 s, both passes.** Those failures are the data that explains
why the on-ground rate looks worse than the in-flight one.

### Finish

```
press  q
```

Exit table must show no `SHORT` rows. If any bin is short, rerun the script and
top up just that distance — the analysis pools every run.

---

## TEST 3 — Session D dataset (webcam only, daylight, ~25 min)

### Before

```
[ ] daylight, curtains open, NO lamps on
[ ] standing in front of bookshelf / furniture / patterned wall
[ ] same webcam as sessions B and C
[ ] measure your distance to the camera  <- you need the number below
```

### Start

```bash
python -m tello_gesture_py.src.gestures.auto_collect_dataset --session D --illumination daylight --background non-uniform --background-description "living room, bookshelf and sofa" --distance-m 1.25 --out_dir data/raw/sessionD --out_csv data/processed/sessionD.csv
```

Edit `--background-description` and `--distance-m` to the truth.

**On distance:** for A, B and C you deliberately varied 0.5–2.0 m to span
scales. If you do the same here, pass the midpoint `--distance-m 1.25` and it
will be recorded alongside the 0.5–2.0 m protocol now stored in
`data/sessions.json` for all four sessions. If instead you hold one fixed
distance, pass that number — but then D differs from A–C in protocol as well as
condition, which weakens the comparison. **Vary it, like the others.**

### Then: almost no keys

Fully automatic, per class:

```
10 s countdown   <- get into the gesture
400 frames       <- ~80 s, hold it (vary your distance across the range)
10 s gap         <- next class
```

Seven classes in order: CENTER, LEFT, RIGHT, UP, DOWN, FORWARD, BACK.

| key | when |
|---|---|
| `q` | stop early — keeps what it captured |

### Then extract features

```bash
python -m tello_gesture_py.src.gestures.images_to_features --dataset data/processed/sessionD.csv --out data/processed/sessionD_features.csv
```

Expect ~2600–2800 kept of 2800. A big drop is a result about the background —
report it, do not recapture.

### Rule

D may score worse than B and C. **That is the design.** Do not adjust lighting,
distance or background to improve its numbers.

---

## When all three are done

```bash
python -m tello_gesture_py.src.gestures.per_session_eval
python -m tello_gesture_py.src.gestures.loso_eval
python -m tello_gesture_py.scripts.analyse_operational_gesture
python -m tello_gesture_py.src.gestures.envelope_eval
python -m tello_gesture_py.src.gestures.build_paper_tables --outdir outputs/paper
python -m tello_gesture_py.src.gestures.build_paper_figures --outdir paper/figures
python -m tello_gesture_py.src.gestures.audit_claims
```

Paste me the output of the last three and I will take it from there.
