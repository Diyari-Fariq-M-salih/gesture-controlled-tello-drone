# Tello Gesture Control — Command Sequence

## 0. Environment
```bash
conda activate tello_drone_mk1
cd tello_gesture_py
```

---

## 1. Run Drone (NO trained model — rule-based)
```bash
python -m tello_gesture.main
```

---

## 2. Collect Gesture Dataset (Webcam)

### Option A — Manual collector
```bash
python -m tello_gesture.gestures.collect_dataset   --labels tello_gesture/gestures/labels_example.json   --out dataset.csv
```

### Option B — Auto collector
```bash
python -m tello_gesture.gestures.auto_collect_dataset
```

---

## 3. Extract MediaPipe Features
```bash
python -m tello_gesture.gestures.images_to_features   --dataset dataset.csv   --out dataset_features.csv
```

---

## 4. Train Gesture Model
```bash
python -m tello_gesture.gestures.train_model   --dataset dataset_features.csv   --labels tello_gesture/gestures/labels_example.json   --out model.joblib   --metrics_out training_metrics.json   --cm_png_out confusion_matrix.png
```

---

## 5. Run Drone WITH trained model
```bash
python -m tello_gesture.main   --model model.joblib   --labels tello_gesture/gestures/labels_example.json
```

---

## 6. Generate Reports (Dataset, Features, Training, Telemetry)
```bash
python .\tello_gesture\utils\reporting.py   --dataset_images dataset.csv   --dataset_features dataset_features.csv   --labels tello_gesture/gestures/labels_example.json   --training_metrics training_metrics.json   --outdir report_out
```

---

## 7. (Optional) Telemetry Only Report
```bash
python .\tello_gesture\utils\reporting.py   --telemetry telemetry_log.csv   --outdir report_out
```
