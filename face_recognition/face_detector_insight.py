import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import os
import math
from typing import Tuple, Union
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
RED_ = (255, 0, 0)  # red
GREEN_ = (0, 255, 0)  # green

assert insightface.__version__>='0.3'

def compare_faces(emb1, emb2, threshold=0.65): # Adjust this threshold according to your usecase.
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold

def recognize_face(face_image: np.ndarray, app: FaceAnalysis) -> list:
    """Recognize face using InsightFace."""
    faces = app.get(face_image)
    return faces

def draw_detections_on_image(rgb_image: np.ndarray, bboxes, first_face_idx) -> np.ndarray:
        """Draws bounding boxes and keypoints on the input image."""
        annotated_image = rgb_image.copy()
        if not bboxes:
            return annotated_image
        for idx, bbox in enumerate(bboxes):
            start_point = bbox[0], bbox[1]
            end_point = bbox[2], bbox[3]
            if idx == first_face_idx:
                caption = f'Authorized: {first_face_idx}'
                text_color = GREEN_
            else:
                caption = 'Intruder'
                text_color = RED_
            cv2.rectangle(annotated_image, start_point, end_point, text_color, 3)
            cv2.putText(annotated_image, caption, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, text_color, FONT_THICKNESS)
        return annotated_image

if __name__ == "__main__":
    # Use insightface for face recognition. 
    # Reference: https://github.com/deepinsight/insightface/blob/master/examples/face_recognition/insightface_app.py 
    # use CUDA if available
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ctx_id = 0
    else:
        print("CUDA not available, using CPU.")
        providers = ['CPUExecutionProvider']
        ctx_id = -1
    # create the insightface
    # doc on available models: https://github.com/deepinsight/insightface/tree/master/model_zoo
    model_pack_name = 'buffalo_sc'
    app = FaceAnalysis(name=model_pack_name, providers=providers)
    app.prepare(ctx_id=ctx_id, det_thresh=0.5) # det_thresh is the detection threshold

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
    
        faces = app.get(frame)
        
        # store embedding of the first detected face only
        if faces and not detected_first_face:
            detected_first_face = True
            # Get the first detection (or you could get the largest one)
            first_face = faces[0]
            bbox = first_face.bbox.astype(int)
            x_min, y_min, x_max, y_max = bbox
            # Crop the face from the frame
            face_image = frame[y_min:y_max, x_min:x_max]
            # Resize the face image to the input size expected by InsightFace (112x112)
            # resized_face = cv2.resize(face_image, (112, 112))
            # new window displaying the cropped and resized image
            cv2.imshow("Resized Face", cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            # Recognize face using InsightFace
            first_face_embedding = first_face.embedding
            first_face_idx = 0
            print("First face detected and embedding stored.")
        # else, compare faces
        else:
            N_faces = len(faces)
            # compare faces
            if N_faces > 1:
                for idx, face in enumerate(faces):
                    face_embedding = face.embedding
                    similarity, is_same = compare_faces(first_face_embedding, face_embedding)
                    if is_same:
                        first_face_idx = idx
                        print("Matching face detected! IDX:", idx)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes = [face.bbox.astype(int) for face in faces]
        print('Authorized face index:', first_face_idx if detected_first_face else "No face detected yet")
        annotated_image = draw_detections_on_image(rgb_frame, bboxes, first_face_idx)
        
        cv2.imshow("Face Detection", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
