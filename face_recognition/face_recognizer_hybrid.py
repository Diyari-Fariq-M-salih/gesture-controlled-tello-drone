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
from insightface.utils import face_align

assert insightface.__version__>='0.3'

class FaceRecognizer:
    """MediaPipe Face Detection + InsightFace recognition."""
    
    def __init__(self, detection_model='blaze_face_short_range.tflite', recognition_model='buffalo_sc', det_thresh=0.65, simil_thresh=0.5):
        """
        Initialize Face Recognizer.
        Args:
            detection_model: MediaPipe detection model filename
            recognition_model: InsightFace model pack name
            det_thresh: MediaPipe detection threshold
            simil_thresh: Similarity threshold for face recognition
        """
        ## 1. Initialize MediaPipe Face Detector
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'models', detection_model)
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=det_thresh)
        self._detector = vision.FaceDetector.create_from_options(options)

        ## 2. Initialize InsightFace Recognizer
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = 0
        else:
            print("CUDA not available, using CPU.")
            providers = ['CPUExecutionProvider']
            ctx_id = -1
        
        # Load only the recognition model
        allowed_modules = ['recognition', 'detection']
        model_pack_name = recognition_model
        app = FaceAnalysis(name=model_pack_name, providers=providers, allowed_modules=allowed_modules)
        app.prepare(ctx_id=ctx_id)
        self.rec_model = app.models['recognition']
        
        # Set similarity threshold
        self.simil_thresh = simil_thresh
        
        # Variables to store authorized face
        self.auth_face_embedding = None
        self.auth_face_bbox = None
        self.initialized_auth_face = False
        self.auth_face_idx = 0
        self.bboxes = []
        
        # Visualization parameters
        self.FONT_SIZE = 1
        self.FONT_THICKNESS = 1
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.frame = None


    def init_auth_face(self, rgb_frame: np.ndarray, detection_result):
        """Initialize the authorized face from the first detection."""
        if len(detection_result.detections) == 1:
            print('Detected one face, storing as authorized face.')
            self.initialized_auth_face = True
            self.auth_face_idx = 0
            
            first_detection = detection_result.detections[0]
            self.auth_face_embedding = self.get_aligned_embedding(rgb_frame, first_detection)
            bbox = self.get_bounding_box_from_detection(first_detection)
            self.auth_face_bbox = bbox
            
            return bbox
        elif len(detection_result.detections) > 1:
            print("Multiple faces detected. Cannot store as authorized. Please ensure only one face is in the frame.")
            return None
        else:
            return None
        
    def recognize(self, image: np.ndarray):
        """
        Detect faces and compare with authorized face.
        Returns the bbox of the authorized face if found, empty list otherwise.
        """
        # Validate input frame
        if image is None or image.size == 0:
            return None
        
        self.frame = image
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces using MediaPipe
        detection_result = self.detect(image)
        
        # If authorized face not initialized, initialize it
        if detection_result.detections and not self.initialized_auth_face:
            return self.init_auth_face(rgb_frame, detection_result)
        
        # Compare faces with authorized face
        elif self.initialized_auth_face:
            self.bboxes = []
            N_faces = len(detection_result.detections)
            
            if N_faces >= 1:
                for idx, face in enumerate(detection_result.detections):
                    bbox = self.get_bounding_box_from_detection(face)
                    self.bboxes.append((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                    
                    # Get embedding for current face
                    other_face_embedding = self.get_aligned_embedding(rgb_frame, face)
                    similarity, is_same = self.compare_faces(self.auth_face_embedding, other_face_embedding)
                    
                    if is_same:
                        self.auth_face_idx = idx
                        self.auth_face_bbox = bbox
                        return self.auth_face_bbox  # Only return if match found
                
                # No match found among detected faces
                return None
            else:
                return None
        
        return None

    def detect(self, bgr: np.ndarray):
        """Detect faces in the input image. Returns list of detections."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = self._detector.detect(image)
        return detection_result

    def compare_faces(self, emb1, emb2):
        """Compare two embeddings using cosine similarity"""
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity, similarity > self.simil_thresh

    def get_annotated_image(self):
        """Draws bounding boxes on the input image."""
        if self.frame is None:
            return None
        
        annotated_image = self.frame.copy()
        
        if not self.bboxes:
            return annotated_image
        
        for idx, bbox in enumerate(self.bboxes):
            start_point = bbox[0], bbox[1]
            end_point = bbox[2], bbox[3]
            
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

    def _normalized_to_pixel_coordinates(
        self, normalized_x: float, normalized_y: float, 
        image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""
        
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    def get_bounding_box_from_detection(self, detection) -> Tuple[int, int, int, int]:
        """Extracts bounding box from a detection."""
        bbox = detection.bounding_box
        return (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
    
    def get_aligned_embedding(self, bgr_frame, detection):
        """Get aligned face embedding using InsightFace."""
        # 1. Extract dimensions
        h, w, _ = bgr_frame.shape
        
        # 2. Re-map MediaPipe keypoints to InsightFace's 5-point format
        # MediaPipe: 0:L-Eye, 1:R-Eye, 2:Nose, 3:Mouth, 4:L-Tragion, 5:R-Tragion
        kps = np.array([[kp.x * w, kp.y * h] for kp in detection.keypoints], dtype=np.float32)
        
        # Format: [LeftEye, RightEye, Nose, LeftMouth, RightMouth]
        aim_kps = np.zeros((5, 2), dtype=np.float32)
        aim_kps[0] = kps[0] # Left Eye
        aim_kps[1] = kps[1] # Right Eye
        aim_kps[2] = kps[2] # Nose
        aim_kps[3] = kps[3] # Use Mouth Center for both corners
        aim_kps[4] = kps[3] 

        # 3. Align and crop to 112x112
        aligned_face = face_align.norm_crop(bgr_frame, landmark=aim_kps, image_size=112)

        # 4. INFERENCE: Pass the BGR image directly. 
        # InsightFace's get_feat handles normalization and transposition internally.
        embedding = self.rec_model.get_feat(aligned_face).flatten()
        
        return embedding