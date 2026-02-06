import numpy as np
import cv2

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

assert insightface.__version__>='0.3'

class FaceRecognizerInsight:
    """InsightFace Face Recognizer wrapper."""
    
    def __init__(self, model_pack_name='buffalo_sc', det_thresh=0.5, simil_thresh=0.65):
        """
        Initialize Face Recognizer.
        Args:
            model_pack_name: Name of the model pack to use (e.g., 'buffalo_sc')
            det_thresh: Detection threshold for face detection
            simil_thresh: Similarity threshold for face recognition (cosine similarity)
        """
        # use CUDA if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0
        else:
            print("CUDA not available, using CPU.")
            providers = ['CPUExecutionProvider']
            ctx_id = -1
        # initialize the insightface app with the specified model pack and providers
        self.app = FaceAnalysis(name=model_pack_name, providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh)
        # set the similarity threshold for face recognition
        self.simil_thresh = simil_thresh
        # variables to store the authorized face embedding and bbox, and a flag to indicate if the authorized face has been initialized
        self.auth_face_embedding = None
        self.auth_face_bbox = None
        self.initialized_auth_face = False
        self.auth_face_idx = 0
        self.bboxes = []
        # visualization parameters
        self.FONT_SIZE = 1
        self.FONT_THICKNESS = 1
        self.RED = (255, 0, 0)  # red
        self.GREEN = (0, 255, 0)  # green
        self.frame = None

    # main function 
    def recognize(self, image: np.ndarray) -> list:
        """Detect faces in the input image and return bbox of the authorized face."""
        # if the authorized face has been initialied, run the main function:
        # compare detected faces with the authorized face and return the bbox of the matching face
        self.frame = image
        if self.initialized_auth_face:
            self.bboxes = []
            faces = self.app.get(image)
            if not faces:
                print("No faces detected in the image.")
                return []
            for idx, face in enumerate(faces):
                # get current face embedding
                face_embedding = face.embedding
                # compare the embedding of the detected face with the authorized face embedding
                similarity, is_same = self.compare_faces(self.auth_face_embedding, face_embedding)
                # if the similarity is above the threshold, return the bbox of the detected face
                # update bboxes (for visualization) to show the bbox of the authorized face in green and others in red
                self.bboxes.append(face.bbox.astype(int)) 
                if is_same:
                    # print("Authorized face detected! IDX:", idx)
                    self.auth_face_bbox = face.bbox.astype(int)
                    # index of the authorized face in the current detections, which can be used to draw bbox in green later
                    self.auth_face_idx = idx
                    # do not return here to allow drawing bbox for all detected faces
            # Return after processing all faces
            if self.auth_face_bbox is not None:
                return self.auth_face_bbox
            print("No authorized face detected in the image.")
            return []
        # else, if the authorized face has not been initialized, initialize it by calling init_auth_face.
        else:
            self.init_auth_face(image)
            return []

    def init_auth_face(self, image: np.ndarray) -> list:
        """Detect the face in the image and store its embedding as the authorized face. Only one face should be in the image for this function to work properly."""
        faces = self.app.get(image)
        if len(faces) == 1:
            print('Detected one face, storing as authorized face.')
            first_face = faces[0]
            # store the embedding and bbox of the first detected face as the authorized face
            self.auth_face_embedding = first_face.embedding
            self.auth_face_bbox = first_face.bbox.astype(int)
            # set the flag to indicate that the authorized face has been initialized
            self.initialized_auth_face = True
            print("Authorized face embedding stored successfully.")
        elif len(faces) > 1:
            print("Multiple faces detected. Cannot store as authorized. Please ensure only one face is in the frame.")
            return []
        else:
            # print("No face detected to store as authorized.")
            return []

    def compare_faces(self, emb1, emb2): # Adjust this threshold according to your usecase.
        """Compare two embeddings using cosine similarity"""
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity, similarity > self.simil_thresh

    def get_annotated_image(self) -> np.ndarray:
            """Draws bounding boxes and keypoints on the input image."""
            # convert the image to RGB for visualization
            annotated_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            if not self.bboxes:
                return annotated_image
            for idx, bbox in enumerate(self.bboxes):
                start_point = bbox[0], bbox[1]
                end_point = bbox[2], bbox[3]
                # Plot in green for authorized face and red for intruders
                if idx == self.auth_face_idx:
                    caption = f'Authorized: {self.auth_face_idx}'
                    text_color = self.GREEN
                else:
                    caption = 'Intruder'
                    text_color = self.RED
                cv2.rectangle(annotated_image, start_point, end_point, text_color, 3)
                cv2.putText(annotated_image, caption, (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_PLAIN, self.FONT_SIZE, text_color, self.FONT_THICKNESS)
            return annotated_image
