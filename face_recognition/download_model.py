import urllib.request
import os

model_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
model_path = os.path.join(os.path.dirname(__file__), 'pose_landmarker.task')

print(f"Downloading model to {model_path}...")
urllib.request.urlretrieve(model_url, model_path)
print("Download complete!")
