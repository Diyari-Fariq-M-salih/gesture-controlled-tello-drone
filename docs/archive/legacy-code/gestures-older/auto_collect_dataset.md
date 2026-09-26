# Gesture Model Pipeline

Use the following commands to collect data, extract features, and train the gesture recognition model.

## 1. Collect Dataset (Webcam Images)

```bash
python -m tello_gesture.gestures.auto_collect_dataset
```

## 2. Extract Hand Landmark Features

```bash
python -m tello_gesture.gestures.images_to_features --dataset dataset.csv --out dataset_features.csv
```

## 3. train using Hand Landmark Features

```bash
python -m tello_gesture.gestures.train_model --dataset dataset_features.csv --labels tello_gesture/gestures/labels_example.json --out model.joblib

```

### 4. Run Gesture Recognition

Use the trained model with the same command as in the previous version of the project.
