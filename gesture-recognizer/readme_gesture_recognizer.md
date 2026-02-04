# Gesture Recognition (MediaPipe + PyTorch)

This project is a **standalone, desktop gesture recognition system** built in Python.

It uses:

- **MediaPipe** → fast, real‑time hand landmark detection
- **PyTorch** → machine‑learning classifier for static hand gestures (frame‑by‑frame)

The output is a **predicted gesture label with confidence**, displayed on screen and printed to the terminal.

This project is intentionally designed as a **foundation** for later extensions (e.g. drone control), while remaining safe, debuggable, and modular.

---

## Project Goals

- Real‑time hand gesture recognition using a webcam
- Clean separation between:
  - perception (hand tracking)
  - learning (gesture classification)
  - decision logic (what to do with the gesture)
- Robust to hand size, position, and scale
- Easy to extend with new gestures

---

## Gesture Set

Current gestures (8 classes):

1. `POINT_FORWARD`
2. `PEACE_BACK` (peace sign with **back of hand** facing camera)
3. `THUMBS_LEFT`
4. `PINKY_RIGHT`
5. `OPEN_UP`
6. `FIST_DOWN`
7. `OKAY_LAND`
8. `LOVE_ROTATE`

> ⚠️ Gestures are **static** (single frame), not dynamic sequences.

---

## High‑Level Architecture

```
Webcam
  ↓
MediaPipe Hands
  ↓
21 hand landmarks (x, y, z)
  ↓
Feature normalization (translation + scale)
  ↓
PyTorch MLP classifier
  ↓
Gesture label + confidence
```

Key idea:

> **MediaPipe finds the hand. PyTorch decides what it means.**

---

## Folder Structure

```
gesture-recognizer/
├─ .venv/                # Virtual environment (recommended)
├─ collect_data.py       # Webcam data collection tool
├─ train.py              # PyTorch training script
├─ infer.py              # Live inference (webcam)
├─ utils.py              # Shared utilities (features, labels)
├─ data/
│   └─ raw/              # Collected training samples (.npz)
├─ models/
│   └─ model.pth         # Trained PyTorch model
└─ README.md
```

---

## Environment Setup

### Create virtual environment (recommended)

From inside the project folder:

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install mediapipe==0.10.8 opencv-python numpy torch torchvision torchaudio
```

Verify:

```bash
python -c "import mediapipe as mp, torch; print(hasattr(mp,'solutions'), torch.__version__)"
```

---

## How It Works (Key Design Choices)

### 1. Landmark Features

- 21 hand landmarks × (x, y, z) → **63 values**
- Wrist landmark is used as origin (translation invariance)
- Scaled by max distance to wrist (scale invariance)
- +1 handedness bit (Left / Right)

**Final feature vector: 64 dimensions**

This allows the model to learn:

- hand orientation
- finger configuration
- left/right differences

---

### 2. Classifier

- Feed‑forward **MLP (Multi‑Layer Perceptron)**
- Trained with cross‑entropy loss
- Dropout used during training for robustness

This is intentionally simple:

- fast inference
- easy debugging
- good performance for static gestures

---

## Usage

### 1️⃣ Collect training data

```bash
python collect_data.py
```

Controls:

- `1..8` → select gesture class
- `S` → save a sample
- `Q` → quit

Recommendations:

- Collect **300–800 samples per gesture**
- Vary:
  - distance from camera
  - slight wrist rotation
  - lighting conditions
- For `PEACE_BACK`, ensure the **back of the hand** faces the camera

---

### 2️⃣ Train the model

```bash
python train.py
```

Output:

- Training + validation accuracy per epoch
- Best model saved to:

```
models/model.pth
```

> ⚠️ Training will warn you if a gesture has **0 samples** — fix this before inference.

---

### 3️⃣ Run live inference

```bash
python infer.py
```

Features:

- Real‑time webcam prediction
- Confidence thresholding
- Terminal output **only when prediction changes**
- On‑screen gesture + confidence display

Press `Q` to quit.

---

## Safety & Robustness Improvements (Important)

Even though this is a desktop project, the following best practices are already applied:

### ✔ Confidence threshold

Low‑confidence predictions are ignored (`UNSURE`).

### ✔ Cooldown

Prevents rapid flickering / spam output.

### ✔ Architecture consistency

Training and inference models **must match exactly**.

### ✔ Explicit gesture set

No hidden magic labels.

---

## Known Limitations

- Static gestures only (no motion‑based gestures yet)
- Requires retraining if gesture set changes
- Accuracy depends heavily on quality & balance of data

---

## Why This Design Scales Well

This project cleanly separates:

- **Perception** → MediaPipe
- **Learning** → PyTorch
- **Decision logic** → your code

This makes it easy to:

- add gesture smoothing
- add temporal models (LSTM / Transformer)
- replace terminal output with:
  - keyboard events
  - robot commands
  - drone SDK commands (later)

---

## Next Possible Extensions

- Temporal smoothing (majority vote over N frames)
- Dynamic gesture recognition (sequences)
- Gesture‑to‑command mapping with safety states
- Export model for deployment

---

## Summary

This mini‑project is a **complete, real‑world gesture recognition pipeline**:

- fast
- modular
- debuggable
- extensible

It is suitable as:

- a learning project
- a research prototype
- the perception layer of a robotics or drone system

---

Happy hacking ✋🚀
