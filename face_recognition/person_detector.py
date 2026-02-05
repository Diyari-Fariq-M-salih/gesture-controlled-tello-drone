import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import os

class PersonDetector:
    """MediaPipe Pose wrapper returning normalized landmarks."""
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), 'pose_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True)
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, bgr: np.ndarray) -> np.ndarray | None:
        """Detect person and return pose landmarks as normalized coordinates."""
        detection_result = self.detect_using_task(bgr)
        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            return None
        # Get first detected person's landmarks
        lm = detection_result.pose_landmarks[0]
        # Convert to numpy array with shape (33, 3) for x, y, z
        arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        return arr
    
    def detect_using_task(self, bgr: np.ndarray) -> vision.PoseLandmarkerResult:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = self._detector.detect(image)
        return detection_result

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
            
            # Draw bounding box using MediaPipe style
            img_height, img_width, _ = annotated_image.shape
            landmarks_array = np.array([[p.x, p.y, p.z] for p in pose_landmarks], dtype=np.float32)
            
            x, y, w, h = self.bounding_box(landmarks_array, img_width, img_height)
            print(f"Person {idx} bounding box: x={x}, y={y}, w={w}, h={h}")
            x, y, w, h = self.compute_bounding_box(landmarks_array, img_width, img_height)
            print(f"Person {idx} bounding box (method 2): x={x}, y={y}, w={w}, h={h}")
            # ensure bounding box is within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_width - x)
            h = min(h, img_height - y)
            
            # Draw rectangle with MediaPipe-style thick border and label
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # Add label background
            label = f"Person {idx}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_image, (x, y - 20), (x + label_size[0], y), (0, 255, 0), -1)
            cv2.putText(annotated_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return annotated_image

    def bounding_box(self, landmarks: np.ndarray, img_width: int, img_height: int) -> tuple[int, int, int, int]:
        """Calculate bounding box from landmarks."""
        x_coords = landmarks[:, 0] * img_width
        y_coords = landmarks[:, 1] * img_height
        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))
        return x_min, y_min, x_max - x_min, y_max - y_min  # x, y, width, height
    
    def compute_bounding_box(self, landmarks: np.ndarray, img_width: int, img_height: int) -> tuple[int, int, int, int]:
        """Compute bounding box from normalized landmarks with padding."""
        # Use key body points for better bounding box
        # Landmark indices: 0=nose, 7=left_ear, 8=right_ear, 11=left_shoulder, 12=right_shoulder, 
        # 23=left_hip, 24=right_hip, 15=left_ankle, 16=right_ankle
        
        # Calculate face radius for padding
        face_rad = 0.5 * (np.linalg.norm(landmarks[0] - landmarks[7]) + np.linalg.norm(landmarks[0] - landmarks[8]))
        
        # Get x boundaries with face padding
        x_min = min(landmarks[7, 0], landmarks[8, 0]) - 0.25 * face_rad  # ears with padding
        x_shoulder = landmarks[[11, 12], 0]
        x_hip = landmarks[[23, 24], 0]
        # Handle case where hips might be outside image
        # if 
        x_max = max(x_shoulder.max(), x_hip.max())
        
        # Get y boundaries
        y_min = min(landmarks[7, 1], landmarks[8, 1]) - 0.5 * face_rad  # ears with padding
        y_shoulder = landmarks[[11, 12], 1].max()
        y_hip = landmarks[[23, 24], 1].max()
        y_max = max(y_hip, y_shoulder)
        # Convert to pixel coordinates
        x = int(x_min * img_width)
        y = int(y_min * img_height)
        w = int((x_max - x_min) * img_width)
        h = int((y_max - y_min) * img_height)
        
        return x, y, w, h
    
if __name__ == "__main__":
    # Test the PersonDetector class
    detector = PersonDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Optimize performance by marking image as not writeable
        frame.flags.writeable = False
        detection_result = detector.detect_using_task(frame)
        frame.flags.writeable = True
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_image = detector.draw_landmarks_on_image(rgb_frame, detection_result)
        cv2.imshow("Pose Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()