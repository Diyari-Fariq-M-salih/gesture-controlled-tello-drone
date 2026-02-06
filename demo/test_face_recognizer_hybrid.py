import cv2
from face_recognition.face_recognizer_hybrid import FaceRecognizer

# Test the FaceRecognizer class
recognizer = FaceRecognizer(detection_model='blaze_face_short_range.tflite', recognition_model='buffalo_sc')

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
    
    # Main recognition logic
    frame.flags.writeable = False
    result = recognizer.recognize(frame)
    frame.flags.writeable = True
    
    # Get annotated image
    annotated_image = recognizer.get_annotated_image()
    cv2.imshow("Face Recognition", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
