

# Doc
**Remark**:     
before use, download models using the script `download_models.py`.


## Test Face recognition
Activate your `.venv` then run in the **root directory**:       
- **Fast approach**:    
-> Uses MediaPipe for detection, InsightFace for recognition        
-> MediaPipe model: `blaze_face_short_range.tflite` (full-range is not released yet).
-> InsightFace model: `buffalo_sc` (smallest, uses FaceNet).
```sh
python -m demo.test_face_recognizer_hybrid
```

- **Accurate approach**, slow if multiple people.       
-> (InsightFace for both detection and recognition). 
-> Model: `buffalo_sc`.
```sh
python -m demo.test_face_recognizer_insight
```
- **Detection only** (MediaPipe, model: `blaze_face_short_range`)
```sh
python -m demo.test_face_detector_mediapipe
```
## OLD
### Person Detector
- Detects a person using Mediapipe `pose_landmarker`. Returns body landmarks.   
- Performs detection Bounding box of the torso.
- Larger bounding box than face detector.

Download mediapipe person detection models.
```sh
cd ~/gesture-controlled-tello-drone/follow_me_mode
\wget -O pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```
**Status**: not used. Face detector would be more useful.

### Face Detector
- Detects all faces in the camera using Mediapipe `face_detector/blaze_face_short_range.tflite`.

- Returns face landmarks + bounding box.   

To use it, first download the mediapipe face detection models.
```sh
cd ~/gesture-controlled-tello-drone/follow_me_mode
\wget -q -O detector.tflite -q https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
```
## Approaches
### Idea 1 (not Robust)
The `face_detector_csrt` Uses OpenCV CSRT tracker by first defining the object in the bounding box (face), then track it using correlation. -> not robust.

### Idea 2


InsightFace splits the work between two distinct types of models:       
1. **Detection Models:** (e.g., `SCRFD` or `RetinaFace`) These scan the whole image to find where faces are and return bounding boxes and landmarks.
2. **Recognition Models:** (e.g., `ArcFace`) These take a *normalized* crop of a face and turn it into a 128 or 512-dimensional vector (embedding).

