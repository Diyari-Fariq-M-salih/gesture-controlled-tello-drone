import numpy as np
import cv2
import sys
from face_recognition.face_recognizer_insight import FaceRecognizerInsight

# Use insightface for face recognition. 
# Reference: https://github.com/deepinsight/insightface/blob/master/examples/face_recognition/insightface_app.py 

# create the insightface
# doc on available models: https://github.com/deepinsight/insightface/tree/master/model_zoo
model_pack_name = 'buffalo_sc'
face_recognizer = FaceRecognizerInsight(model_pack_name=model_pack_name)
detected_first_face = False

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Get the bounding box of the authorized face in the current frame
    bbox = face_recognizer.recognize(frame)
    ## Two modes:
    ## 1. Gesture control: 
    ## detect gestures only if the authorized face is detected in the frame, 
    ## otherwise skip detection to save computation and avoid false positives.
    
    # if bbox:
    #     # run gesture control code here, using the bbox of the authorized face for gesture
    
    ## 2. Face tracker (follow me mode):   
    ## track the authorized face in the frame and draw bbox around it, without gesture control. 
    ## this can be used as a "follow me" mode where the system only needs to track the authorized user without recognizing gestures.
    ## IMPORTANT: modify the class FaceFollower to take the bbox of the authorized face as input instead of the current frame
    
    # face_follower = FaceFollower() # create an instance of the face follower class, which takes the bbox of the authorized face as input and tracks it in real time.
    # cmd, dbg = face_follower.update(bbox) # update the follower with the current frame and the bbox of the authorized face to draw bbox and track the face in real time.

    # OPTIONAL: for visualization, draw bbox around the detected authorized face in green and other faces (if any) in red.
    annotated_image = face_recognizer.get_annotated_image()
    
    cv2.imshow("Face Detection", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
