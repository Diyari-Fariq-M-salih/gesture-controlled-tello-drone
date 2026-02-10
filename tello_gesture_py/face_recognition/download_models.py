import urllib.request
import os

# Download mediapipe model for face detection
# models = filename: url
models = {
    'pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
    'blaze_face_short_range.tflite': 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
}

for filename, url in models.items():
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, model_path)
    print(f"✓ Downloaded to {model_path}")

print("All models downloaded successfully!")

