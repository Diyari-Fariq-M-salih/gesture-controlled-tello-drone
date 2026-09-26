# Flight day protocol

## Setup (once)

See [learnings/environment.md](learnings/environment.md): CPython 3.12 is required, and how to rebuild `.venv`.

## Every flight

Power the Tello, join its Wi-Fi, then launch with a **run id**:

```bash
python -m tello_gesture_py.src.main --run-id drone_d1-latency-hover --note "battery 1, latency hover"
```

Every launch writes its own directory — nothing is ever overwritten:

```
outputs/runs/20260828-141530_drone_d1-latency-hover/
    manifest.json     config, git SHA, library versions, host, ablations
    telemetry.csv     drone state at 1 Hz
    decisions.csv     mode / command / reason / LLM explanation
    perf.csv          per-frame stage latencies, frame age, fps
    perf_events.csv   rc_send, llm_explanation, failsafe, emergency
    scenarios.csv     trial outcomes with preconditions
```

Name runs `<type>_<what>` so they read well later: `drone_d1-latency-hover`,
`drone_d2-trials`, `drone_d2-ablation-no-hysteresis`, `webcam_d3-spoof-print`.
The controller adds the timestamp. Type: `drone`, `webcam`, `capture`, `latency`.

## Keys

| Key | Action |
|---|---|
| `t` / `l` / `e` | takeoff / land / emergency stop |
| `p` / `o` | start-stop FaceID enrollment / clear template |
| `1`–`8` | open a scenario trial |
| `y` / `n` | resolve open trial pass / fail |
| `x` | **discard** an open trial (mis-started or interrupted) |
| `w a s d` `r f` `j k` | manual override: left/right, forward/back, up/down, yaw |
| `m` | sync marker for an external video (clap at the beep) |
| `c` / `k` | cued runs: pause-resume the schedule / skip a cue (`k` also yaws; harmless under `--no-actuate`) |
| `v` / `b` | landmark overlay / pose on every frame (association) |
| `i` / `h` | show-hide the LLM text / key help on screen |
| `q` | quit and export |

The HUD shows the run name, active ablations and any open trial, so a screen
recording is self-labelling — you can score trials from video afterwards
without a separate notebook.

## The four run types

**1. Latency.** One battery, stationary hover, operator enrolled and in frame.

```bash
python -m tello_gesture_py.src.main --run-id drone_d1-latency-hover --note "latency hover"
```

**2. Scenario trials.** Open a trial with `1`–`8`, perform the behaviour,
resolve with `y`/`n`. Target N≥20 per scenario.

```bash
python -m tello_gesture_py.src.main --run-id drone_d2-trials --note "battery 2"
```

**3. Ablations.** One switch per flight, paired against a normal run.

```bash
python -m tello_gesture_py.src.main --run-id drone_d2-ablation-no-hysteresis --no-hysteresis
python -m tello_gesture_py.src.main --run-id drone_d2-ablation-no-gate --no-identity-gate
```

**4. Battery failsafe.** No special flag — keep flying to the 15% threshold
instead of landing manually. The failsafe fires, logs a
`battery_failsafe_land` event, and you press `8` then `y`. One free trial per
battery.

## Bench check (no propellers turning)

```bash
python -m tello_gesture_py.src.main --run-id drone_bench --no-fly
```

`--no-fly` blocks takeoff entirely, so you can verify video, detection,
enrollment and logging with the aircraft on the desk.

## After each battery

```bash
python -m tello_gesture_py.src.gestures.summarize_perf --run outputs/runs/<dir> --markdown
```

Stage latencies are reported **conditioned on the stage actually running**,
with the duty cycle alongside. That pair is what goes in the paper. Do not
average a throttled stage over all frames — that is what produced the
impossible 0.47 ms face-detection figure in the current draft.
