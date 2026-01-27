# Tello Gesture Control (Python)

## 1. Connect to the Tello

1. Power on the Tello
2. Connect your PC to the drone Wi-Fi (`TELLO-XXXXXX`)
3. Close the Tello mobile app

---

## 2. Run (NO trained model)

This uses **rule-based gestures** (no ML model required).

```bash
python -m tello_gesture.main
```

### What happens

- Uses the **Tello camera** for gesture detection
- Opens a video window
- Sends RC commands at ~10 Hz
- Logs telemetry once per second
- Saves `telemetry_log.csv` on exit

---

## Controls

> The OpenCV window must be focused.

### Flight

| Key     | Action             |
| ------- | ------------------ |
| `t`     | Takeoff            |
| `l`     | Land               |
| `e`     | Emergency stop     |
| `q`     | Quit (lands first) |
| `SPACE` | Stop movement      |

### Modes

| Key | Action                         |
| --- | ------------------------------ |
| `m` | Toggle gesture / keyboard mode |

### Keyboard mode

| Key     | Action           |
| ------- | ---------------- |
| `w / s` | Forward / Back   |
| `a / d` | Left / Right     |
| `r / f` | Up / Down        |
| `j / k` | Yaw Left / Right |

---

## Gesture Control (No Model)

Default rule-based logic:

- **LEFT / RIGHT / UP / DOWN** → index finger direction
- **FORWARD / BACK** → hand size change (closer / farther)

This works immediately but forward/back is less stable than a trained model.

---

## 3. Train Your Own Gesture Model (Recommended)

Training improves accuracy and stability.

### Step 1 — Collect dataset (Webcam)

This uses your **PC webcam**, not the drone.

```bash
python -m tello_gesture.gestures.collect_dataset   --labels tello_gesture/gestures/labels_example.json   --out dataset.csv
```

- Show a gesture
- Press the **number key** assigned to that gesture
- Press `q` to quit

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

### Step 2 — Train the model

```bash
python -m tello_gesture.gestures.train_model   --dataset dataset.csv   --labels tello_gesture/gestures/labels_example.json   --out model.joblib
```

Output:

- Accuracy report
- Confusion matrix
- Trained model file: `model.joblib`

---

## 4. Run WITH trained model

```bash
python -m tello_gesture.main   --model model.joblib   --labels tello_gesture/gestures/labels_example.json
```

- Gesture recognition now uses the trained model
- Still uses the **Tello camera**
- Forward/back motion is significantly more stable

---

## Telemetry Logging

- Saved as `telemetry_log.csv`
- Logged at **1 Hz**
- Includes battery, height, yaw, and velocities

---

## Safety Tips

- Test **keyboard mode first**
- Train gestures before flying
- Keep yaw on keyboard only (recommended)
- Be ready to press `e` for emergency stop

---

## Summary

| Mode               | Camera Used | Purpose               |
| ------------------ | ----------- | --------------------- |
| Control (no model) | Tello       | Quick testing         |
| Control (trained)  | Tello       | Stable gesture flight |
| Dataset collection | Webcam      | Safe training         |
| Keyboard mode      | None        | Debug & safety        |
