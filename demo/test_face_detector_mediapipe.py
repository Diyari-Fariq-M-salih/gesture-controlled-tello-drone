import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import os
import math
from typing import Tuple, Union

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


class FaceDetectorMediaPipe:
    """MediaPipe Face Detection wrapper."""
    
    def __init__(self, model_selection=0, min_detection_confidence=0.5):
        """
        Initialize Face Detector.
        Args:
            model_selection: 0 for short-range (within 2 meters), 1 for full-range (not released yet)
            min_detection_confidence: Minimum confidence threshold
        """
        # Select model based on model_selection parameter
        model_dir = 'face_recognition/models'
        model_path = os.path.join(model_dir, 'blaze_face_short_range.tflite')
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
    
    def resize_face_image(self, image: np.ndarray, detection) -> np.ndarray:
        """resize to 160x160 for face recognition model input (InsightFace)"""
        bbox = self.get_bounding_box_from_detection(detection)
        x, y, w, h = bbox
        cropped_img = image[y:y+h, x:x+w]
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


if __name__ == "__main__":
    # Test the FaceDetector class
    detector = FaceDetectorMediaPipe()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
    
        # Optimize performance
        frame.flags.writeable = False
        detection_result = detector.detect(frame)
        frame.flags.writeable = True
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_image = detector.draw_detections_on_image(rgb_frame, detection_result)
        cv2.imshow("Face Detection", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        # new window displaying the cropped and resized image
        if detection_result.detections:
            # Get the first detection (or you could get the largest one)
            first_detection = detection_result.detections[0]
            resized_face = detector.resize_face_image(rgb_frame, first_detection)
            cv2.imshow("Resized Face", cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
