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

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
RED_ = (255, 0, 0)  # red
GREEN_ = (0, 255, 0)  # green

assert insightface.__version__>='0.3'

class FaceDetector:
    """MediaPipe Face Detection wrapper."""
    
    def __init__(self, model_selection=0, min_detection_confidence=0.65):
        """
        Initialize Face Detector.
        Args:
            model_selection: 0 for short-range (within 2 meters), 1 for full-range (not released yet)
            min_detection_confidence: Minimum confidence threshold
        """
        # Select model based on model_selection parameter
        model_path = os.path.join(os.path.dirname(__file__), 'blaze_face_short_range.tflite')
        # Create base options with the model path
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence)
        # Create the face detector instance
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, bgr: np.ndarray):
        """Detect faces in the input image. Returns list of detections."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = self._detector.detect(image)
        return detection_result

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
    
    def resize_face_image(self, image: np.ndarray, detection, padding=0.2) -> np.ndarray:
        """resize to 160x160 for face recognition model input (InsightFace)"""
        bbox = self.get_bounding_box_from_detection(detection)
        x, y, w, h = bbox
        
        # Add padding to ensure InsightFace can detect the face
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate padded coordinates with boundary checks
        img_h, img_w = image.shape[:2]
        x_min = max(0, x - pad_w)
        y_min = max(0, y - pad_h)
        x_max = min(img_w, x + w + pad_w)
        y_max = min(img_h, y + h + pad_h)
        
        cropped_img = image[y_min:y_max, x_min:x_max]
        resized_img = cv2.resize(cropped_img, (160, 160))
        return resized_img

    # optional for display
    def draw_detections_on_image(self, rgb_image: np.ndarray, detection_result) -> np.ndarray:
        """Draws bounding boxes and keypoints on the input image."""
        annotated_image = rgb_image.copy()
        height, width, _ = rgb_image.shape

        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

            # Draw keypoints
            for keypoint in detection.keypoints:
                keypoint_px = self._normalized_to_pixel_coordinates(
                    keypoint.x, keypoint.y, width, height)
                if keypoint_px:
                    color, thickness, radius = (0, 255, 0), 2, 2
                    cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name if category.category_name else ''
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(annotated_image, result_text, text_location, 
                       cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        return annotated_image

    # --- Use the cropped image for face recognition with InsightFace ---
    def get_face_embedding(self, cropped_face_image: np.ndarray, rec_model) -> np.ndarray:
        """Get face embedding directly using InsightFace recognition model."""
        # Resize to model input size (112x112 for most InsightFace models)
        face_img = cv2.resize(cropped_face_image, (112, 112))
        # Normalize and prepare for model
        face_img = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0).astype(np.float32)
        face_img = (face_img - 127.5) / 127.5
        # Get embedding using the model's session
        embedding = rec_model.get_feat(face_img).flatten()
        # input_name = rec_model.input_names[0]
        # outputs = rec_model.session.run(None, {input_name: face_img})
        # embedding = outputs[0][0]
        if embedding is None:
            raise ValueError("Failed to get embedding from the recognition model.")
        return embedding
    
    def get_aligned_embedding(self, bgr_frame, detection, rec_model):
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
        embedding = rec_model.get_feat(aligned_face).flatten()
        
        return embedding

def compare_faces(emb1, emb2, threshold=0.5): # Adjust this threshold according to your usecase.
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold

def draw_detections(rgb_image: np.ndarray, bboxes, first_face_idx) -> np.ndarray:
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
    # Test the FaceDetector class
    detector = FaceDetector()
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
    
    # Load only the recognition model
    allowed_modules = ['recognition', 'detection']  # Load detection module as well since we need it for cropping, but we won't use it for recognition
    model_pack_name = 'buffalo_sc'
    app = FaceAnalysis(name=model_pack_name, providers=providers, allowed_modules=allowed_modules)
    app.prepare(ctx_id=ctx_id)
    rec_model = app.models['recognition']

    detected_first_face = False
    first_face_idx = 0

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
    
        bboxes = []
        # Optimize performance
        frame.flags.writeable = False
        detection_result = detector.detect(frame)
        frame.flags.writeable = True
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_image = detector.draw_detections_on_image(rgb_frame, detection_result)
        cv2.imshow("Face Detection", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # first detected face only
        if detection_result.detections and not detected_first_face:
            detected_first_face = True
            first_face_idx = 0
            # Get the first detection (or you could get the largest one)
            first_detection = detection_result.detections[0]
            # resized_face = detector.resize_face_image(rgb_frame, first_detection, padding=0.2)
            # new window displaying the cropped and resized image
            # cv2.imshow("Authorized Person (Resized Face)", cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR))
            # Get embedding directly from recognition model
            first_face_embedding = detector.get_aligned_embedding(rgb_frame, first_detection, rec_model)
            print("First face detected and embedding stored.")
        # else, compare faces
        elif detected_first_face:
            N_faces = len(detection_result.detections)
            bboxes = []
            # compare faces
            if N_faces >= 1:
                for idx, face in enumerate(detection_result.detections):
                    resized_face = detector.resize_face_image(rgb_frame, face, padding=0.2)
                    bbox = detector.get_bounding_box_from_detection(face)
                    bboxes.append((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
                    # get embedding directly
                    # other_face_embedding = detector.get_face_embedding(resized_face, rec_model)
                    other_face_embedding = detector.get_aligned_embedding(rgb_frame, face, rec_model)
                    similarity, is_same = compare_faces(first_face_embedding, other_face_embedding)
                    if is_same:
                        # update the index (mainly for visualization)
                        first_face_idx = idx
                        # print(f"Matching face detected! Similarity: {similarity:.4f}")
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 
            # print('Authorized face index:', first_face_idx if detected_first_face else "No face detected yet")
            annotated_image = draw_detections(rgb_frame, bboxes, first_face_idx)
            cv2.imshow("Face Recognition", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
