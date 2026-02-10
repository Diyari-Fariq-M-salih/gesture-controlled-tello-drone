# Tello Drone – Tunable Parameters Reference

This document lists **all key parameters you can safely tune** in the provided files to adjust
face detection, hand detection, gesture sensitivity, mode switching, and control behavior.

---

## 1. Face Detection & Face Following (`face_follow.py`)

### MediaPipe Face Detection Sensitivity

Controls how easily a face is detected.

- `min_detection_confidence = 0.7`  
  ↓ Lower → detects more faces (more false positives)  
  ↑ Higher → stricter detection (may miss faces)

- `model_selection = 0`
  - `0` = short-range, faster
  - `1` = better distance robustness, slightly slower

---

### Face Detection Lifetime

Controls how long a face is considered “present” after last detection.

- `lost_timeout_s = 0.7`  
  (Overridden to `2.0` in `controller.py`)

  ↑ Higher → face mode persists longer on dropped frames  
  ↓ Lower → faster loss of face mode

---

### Detection Resolution & Frequency (Performance vs Accuracy)

- `detect_w = 256`
- `detect_h = 192`  
  ↑ Increase → better detection, slower  
  ↓ Decrease → faster, less reliable

- `detect_every_n = 5`  
  Run face detection every N frames  
  ↑ Higher → faster, more laggy  
  ↓ Lower → smoother, heavier CPU

- `control_hz = 15.0`  
  Max RC update rate in face-follow mode

---

### Face-Follow Control Behavior

- `target_area_frac = 0.075`  
  Desired face size in frame  
  ↑ Higher → drone moves closer  
  ↓ Lower → drone stays farther

- Proportional gains (movement aggressiveness):
  - `kp_yaw`
  - `kp_ud`
  - `kp_fb`

- Output limits:
  - `max_yaw`
  - `max_ud`
  - `max_fb`

- Deadbands (ignore small errors):
  - `deadband_px`
  - `deadband_area`

- Face size smoothing:
  - `area_ema_alpha`  
    ↑ Faster reaction, noisier  
    ↓ Smoother, slower

---

## 2. Hand Detection (`hand_gesture.py`)

### MediaPipe Hands Sensitivity

- `min_detection_confidence = 0.6`  
  ↓ More hands detected, more noise  
  ↑ Fewer false positives, more misses

- `min_tracking_confidence = 0.6`  
  ↑ More stable tracking  
  ↓ Easier to lose hand

- `model_complexity = 1`
  - `0` = fastest, least accurate
  - `1` = balanced
  - `2` = slowest, most accurate

- `max_num_hands = 1`  
  Can be increased to `2` (CPU cost)

---

## 3. Gesture Classification Thresholds

(`config.py` → `RuleBasedGesture`)

### Direction & Scale Thresholds

- `dir_thr = 0.10`  
  ↓ Easier LEFT/RIGHT/UP/DOWN  
  ↑ More stable, less responsive

- `scale_thr = 0.18`  
  ↓ Easier FORWARD/BACK (risk false triggers)  
  ↑ Safer, harder to trigger

- `ema_alpha = 0.35`  
  ↑ Faster reaction, noisier  
  ↓ Smoother, more latency

---

## 4. Mode Switching & Stability

(`controller.py`, `mode_manager.py`)

### Detection Throttling

- `_hand_every_n = 2`  
  Run hand detection every N frames

- `_face_every_n = 4`  
  Run face observe every N frames

---

### Streak Gating (False Positive Control)

- `_hand_streak_on = 2`  
  Required consecutive hand detections

- `_face_streak_on = 1`

- `_gesture_fb_streak_on = 2`  
  FORWARD/BACK must persist N frames

---

### Mode Hysteresis & Autonomy

- `mode_hold_s = 1.2`  
  Prevents rapid mode flapping

- `hand_release_s = 0.8`
- `face_release_s = 0.8`

- `nohuman_search_s = 10.0`  
  Time before search mode

- `search_duration_s = 5.0`
- `search_cooldown_s = 10.0`

- `_search_yaw_cmd = 80`  
  Search spin aggressiveness

- `battery_land_pct = 15`  
  Safety landing threshold

---

## 5. RC Command Feel

(`config.py`)

- `rc_speed = 30`  
  Base motion intensity

- `rc_deadband = 6`  
  Too high → drone may hover forever  
  Must be **less than rc_speed**

- `rc_limit = 100`  
  Command clamp

- `rc_dt = 0.1`  
  RC send interval (10 Hz)

---

## 6. Video Stream Robustness

(`controller.py`)

Stream URL parameters:

- `fifo_size = 5000000`  
  ↑ More buffering (less stutter, more latency)

- `overrun_nonfatal = 1`  
  Prevents stream crash on packet loss

---

## Recommended First Tweaks

If tuning from scratch, start here:

1. Face detection: `min_detection_confidence`
2. Face resolution: `detect_w`, `detect_h`
3. Gesture sensitivity: `dir_thr`, `scale_thr`
4. Mode stability: `mode_hold_s`, `hand_release_s`
5. Control issues: `rc_deadband` vs `rc_speed`

---

End of reference.
