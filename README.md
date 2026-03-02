# Tello Gesture Control System (Python)

---

## 🎥 Demo

### Diyari Auth ID

![Face Follow Demo](media/gifs/face_follow.gif)

---

### ilyas Diyari Auth ID

![Search Mode Demo](media/gifs/search_mode.gif)

# 1. Overview

This project is a modular Python re‑implementation of a Tello SDK
controller with:

- UDP command interface (8889)
- Telemetry listener (8890)
- Video stream (11111)
- MediaPipe hand landmarks
- Optional face-follow mode
- Rule-based gesture control
- Trained ML gesture classification
- Keyboard fallback
- 1 Hz telemetry logging
- Experiment reporting utilities

The system supports:

- Immediate rule-based gesture flight
- Dataset collection (webcam)
- Feature extraction
- Model training
- Deployment with trained classifier
- Report generation
- Tunable real-time control parameters

---

---

## Repo layout note (current)

- `data/` labels + datasets
- `models/` trained artifacts
- `outputs/` generated CSVs/plots/reports
- `tello_gesture_py/src/` Python modules

---

# 2. Installation

## Create Environment

```bash
python -m venv .venv
```

### Windows

```bash
.\.venv\Scripts\activate
```

### Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 3. Running the Drone

## Connect to the Tello

1.  Power on the Tello
2.  Connect PC to drone Wi-Fi (TELLO-XXXXXX)
3.  Close the Tello mobile app

---

## Run WITHOUT Trained Model (Rule-Based)

```bash
python -m tello_gesture_py.src.main
```

Behavior:

- Uses Tello camera
- Opens OpenCV window
- Sends RC commands (\~10 Hz)
- Logs telemetry at 1 Hz
- Saves outputs/telemetry_log.csv on exit

---

## Controls (Window must be focused)

### Flight

t : takeoff\
l : land\
e : emergency stop\
q : quit (lands first)\
SPACE : stop movement

### Modes

m : toggle gesture / keyboard mode

### Keyboard Mode Movement

w / s : forward / back\
a / d : left / right\
r / f : up / down\
j / k : yaw left / right

---

# 4. Gesture Control (Rule-Based)

Default logic:

- LEFT / RIGHT / UP / DOWN → index finger direction (tip vs MCP)
- FORWARD / BACK → hand scale change (bbox area)

Forward/back is less stable without trained model.

---

# 5. Dataset + Training Pipeline

## 5.1 Collect Dataset (Webcam)

> Put your label map in `data/labels/labels_example.json` (you can start by copying `tello_gesture_py/src/gestures/labels_example.json`).

Manual collector:

```bash
python -m tello_gesture_py.src.gestures.collect_dataset \
  --labels data/labels/labels_example.json \
  --out outputs/dataset.csv
```

Auto collector:

```bash
python -m tello_gesture_py.src.gestures.auto_collect_dataset \
  --labels data/labels/labels_example.json \
  --outdir data/raw
```

Example labels:

```json
{
  "1": "LEFT",
  "2": "RIGHT",
  "3": "UP",
  "4": "DOWN",
  "5": "FORWARD",
  "6": "BACK"
}
```

---

## 5.2 Extract MediaPipe Features

```bash
python -m tello_gesture_py.src.gestures.images_to_features \
  --dataset outputs/dataset.csv \
  --out outputs/dataset_features.csv
```

---

## 5.3 Train Gesture Model

```bash
python -m tello_gesture_py.src.gestures.train_model \
  --dataset outputs/dataset_features.csv \
  --labels data/labels/labels_example.json \
  --out models/experiments/model.joblib \
  --metrics_out outputs/training_metrics.json \
  --cm_png_out outputs/confusion_matrix.png
```

Outputs:

- Accuracy report
- Confusion matrix
- Trained model file
- Metrics JSON

---

## 5.4 Run WITH Trained Model

```bash
python -m tello_gesture_py.src.main   --model model.joblib   --labels tello_gesture/gestures/labels_example.json
```

Forward/back motion becomes significantly more stable.

---

# 6. Reporting

## Full Experiment Report

```bash
python tello_gesture_py/src/utils/reporting.py \
  --dataset_images outputs/dataset.csv \
  --dataset_features outputs/dataset_features.csv \
  --labels data/labels/labels_example.json \
  --training_metrics outputs/training_metrics.json \
  --outdir outputs/experiment_runs
```

## Telemetry Only

```bash
python tello_gesture_py/src/utils/reporting.py \
  --telemetry outputs/outputs/telemetry_log.csv \
  --outdir outputs/experiment_runs
```

---

# 7. Telemetry Logging

- Saved as outputs/telemetry_log.csv
- Decisions saved as outputs/telemetry_log_decisions.csv
- Logged at 1 sample/sec
- Includes battery, height, yaw, velocities

---

# 8. Face Detection & Face Following Parameters

## MediaPipe Face Detection

- min_detection_confidence = 0.7
- model_selection = 0

## Lifetime

- lost_timeout_s = 0.7 (overridden to 2.0 in controller.py)

## Detection Resolution

- detect_w = 256
- detect_h = 192
- detect_every_n = 5
- control_hz = 15.0

## Behavior

- target_area_frac = 0.075
- kp_yaw, kp_ud, kp_fb
- max_yaw, max_ud, max_fb
- deadband_px, deadband_area
- area_ema_alpha

---

# 9. Hand Detection Parameters

- min_detection_confidence = 0.6
- min_tracking_confidence = 0.6
- model_complexity = 1
- max_num_hands = 1

---

# 10. Gesture Thresholds (RuleBasedGesture)

- dir_thr = 0.10
- scale_thr = 0.18
- ema_alpha = 0.35

---

# 11. Mode Switching & Stability

## Detection Throttling

- \_hand_every_n = 2
- \_face_every_n = 4

## Streak Gating

- \_hand_streak_on = 2
- \_face_streak_on = 1
- \_gesture_fb_streak_on = 2

## Hysteresis & Autonomy

- mode_hold_s = 1.2
- hand_release_s = 0.8
- face_release_s = 0.8
- nohuman_search_s = 10.0
- search_duration_s = 5.0
- search_cooldown_s = 10.0
- \_search_yaw_cmd = 80
- battery_land_pct = 15

---

# 12. RC Command Feel

- rc_speed = 30
- rc_deadband = 6 (must be \< rc_speed)
- rc_limit = 100
- rc_dt = 0.1

---

# 13. Video Stream Robustness

- fifo_size = 5000000
- overrun_nonfatal = 1

---

# 14. Recommended First Tweaks

1.  Face detection: min_detection_confidence
2.  Face resolution: detect_w, detect_h
3.  Gesture sensitivity: dir_thr, scale_thr
4.  Mode stability: mode_hold_s, hand_release_s
5.  Control feel: rc_deadband vs rc_speed

---

# 15. Summary

Mode Camera Used Purpose

---

Control (no model) Tello Quick testing
Control (trained) Tello Stable gesture flight
Dataset collection Webcam Safe training
Keyboard mode None Debug & safety

---

End of document.
