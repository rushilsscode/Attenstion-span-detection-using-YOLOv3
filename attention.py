import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("C:/Users/Ayush/Desktop/yolov3_project/runs/detect/train11/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)  # 0 for default camera
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Window setup
cv2.namedWindow("Student Attention Tracker", cv2.WINDOW_NORMAL)
cv2.moveWindow("Student Attention Tracker", 100, 100)

# Variables for tracking state and time
previous_state = "Neutral"
state_start_time = time.time()
state_durations = {"Focused": 0, "Neutral": 0, "Distracted": 0, "No Face": 0}  # ✅ Added "No Face"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO on the frame
    results = model(frame)
    
    attention_status = "Neutral"
    detected = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            cls = int(box.cls[0])  # Get class index
            conf = float(box.conf[0])  # Confidence score
            detected = True

            # Assign labels based on class index (modify as per your dataset)
            if cls == 0:  # Assuming 0 = Attentive, 1 = Distracted
                label = "Attentive"
                color = (0, 255, 0)  # Green
                attention_status = "Focused"
            else:
                label = "Distracted"
                color = (0, 0, 255)  # Red
                attention_status = "Distracted"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # If no face detected, mark as "No Face"
    if not detected:
        attention_status = "No Face"

    # Update time spent in each state
    current_time = time.time()
    elapsed_time = current_time - state_start_time
    state_durations[previous_state] += elapsed_time  # ✅ Now "No Face" is valid
    state_start_time = current_time
    previous_state = attention_status

    # Display attention status and statistics
    cv2.putText(frame, f"Attention: {attention_status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Focused: {state_durations['Focused']:.1f}s", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Neutral: {state_durations['Neutral']:.1f}s", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Distracted: {state_durations['Distracted']:.1f}s", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"No Face: {state_durations['No Face']:.1f}s", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show real-time window
    cv2.imshow("Student Attention Tracker", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
