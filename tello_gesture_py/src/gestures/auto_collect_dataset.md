# Gesture Model Pipeline

Use the following commands to collect data, extract features, and train the gesture recognition model.

## 1. Collect Dataset (Webcam Images)

```bash
set PYTHONPATH=./tello_gesture_py/src
python -m gestures.auto_collect_dataset
```

## 2. Extract Hand Landmark Features

```bash
set PYTHONPATH=./tello_gesture_py/src
python -m gestures.images_to_features --dataset data/raw/dataset.csv --out data/processed/dataset_features.csv
```

## 3. train using Hand Landmark Features

```bash
set PYTHONPATH=./tello_gesture_py/src
python -m gestures.train_model --dataset data/processed/dataset_features.csv --labels data/labels/labels_example.json --out models/experiments/model.joblib

```

### 4. Run Gesture Recognition

Use the trained model with the same command as in the previous version of the project.
