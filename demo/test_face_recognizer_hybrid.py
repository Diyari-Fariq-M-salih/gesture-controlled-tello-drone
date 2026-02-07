import cv2
import time
from face_recognition.face_recognizer_hybrid import FaceRecognizer

# Test the FaceRecognizer class
recognizer = FaceRecognizer(detection_model='blaze_face_short_range.tflite', recognition_model='buffalo_sc')

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# FPS tracking variables
prev_time = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Main recognition logic
    frame.flags.writeable = False
    result = recognizer.recognize(frame)
    frame.flags.writeable = True
    
    # Get annotated image
    annotated_image = recognizer.get_annotated_image()
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time
    
    # Convert to BGR for display
    display_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    # Draw FPS on the image
    cv2.putText(display_image, f'FPS: {fps:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Face Recognition", display_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
