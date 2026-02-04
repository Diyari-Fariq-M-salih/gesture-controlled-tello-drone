import urllib.request
import os

models = {
    'blaze_face_short_range.tflite': 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
    'blaze_face_full_range.tflite': 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite'
}

for filename, url in models.items():
    model_path = os.path.join(os.path.dirname(__file__), filename)
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, model_path)
    print(f"✓ Downloaded to {model_path}")

print("All models downloaded successfully!")
