# Tello Gesture Control (Python)

Python re-implementation of a modular Tello SDK controller:

- UDP commands (8889)
- Telemetry listener (8890)
- Video stream (11111)
- MediaPipe hands landmarks
- Controller loop:
  - Gesture-based XYZ translation (lr/fb/ud via RC)
  - Keyboard fallback
  - 1 Hz telemetry logging to CSV

## Install

python -m venv .venv

# Windows:

.\.venv\Scripts\activate

# Linux:

source .venv/bin/activate

pip install -r requirements.txt

## Run

1. Connect PC to Tello Wi-Fi (TELLO-XXXXXX)
2. Power on drone
3. Run:
   python -m tello_gesture.main

## Keys (OpenCV window must be focused)

t : takeoff
l : land
e : emergency
q : quit (tries to land)
m : toggle keyboard mode (default is gesture mode)
SPACE : stop movement (zero RC)

Keyboard movement (in keyboard mode):
w/s : forward/back
a/d : left/right
r/f : up/down
j/k : yaw left/right

## Gesture mapping (default rule-based)

- LEFT/RIGHT/UP/DOWN from index direction (tip vs mcp)
- FORWARD/BACK estimated from hand scale (bbox area change)

## Telemetry logging

On exit writes telemetry_log.csv (1 sample/sec).

## Dataset + Training (optional)

1. Collect:
   python -m tello_gesture.gestures.collect_dataset --labels tello_gesture/gestures/labels_example.json --out dataset.csv

2. Train:
   python -m tello_gesture.gestures.train_model --dataset dataset.csv --labels tello_gesture/gestures/labels_example.json --out model.joblib

3. Run with trained model:
   python -m tello_gesture.main --model model.joblib --labels tello_gesture/gestures/labels_example.json
