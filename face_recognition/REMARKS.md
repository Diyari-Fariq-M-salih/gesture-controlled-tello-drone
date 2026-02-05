

# Remarks
## Person Detector
- Detects a person using Mediapipe `pose_landmarker`. Returns body landmarks.   
- Performs detection Bounding box of the torso.
- Larger bounding box than face detector.

Download mediapipe person detection models.
```sh
cd ~/gesture-controlled-tello-drone/follow_me_mode
\wget -O pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```
**Status**: not used. Face detector would be more useful.

## Face Detector
- Detects all faces in the camera using Mediapipe `face_detector/blaze_face_short_range.tflite`.

- Returns face landmarks + bounding box.   

To use it, first download the mediapipe face detection models.
```sh
cd ~/gesture-controlled-tello-drone/follow_me_mode
\wget -q -O detector.tflite -q https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
```
### Idea 1 (not Robust)
The `face_detector_csrt` Uses OpenCV CSRT tracker by first defining the object in the bounding box (face), then track it using correlation. -> not robust.

### Idea 2