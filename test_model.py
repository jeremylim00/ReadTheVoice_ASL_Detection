
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. LOAD RANDOM FOREST MODEL
# ==========================================

model = joblib.load("asl_model.pkl")

print("Random Forest model loaded!")
print("Classes:", model.classes_)

# ==========================================
# 2. INITIALIZE MEDIAPIPE TASKS
# ==========================================

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = vision.HandLandmarker.create_from_options(options)

print("MediaPipe HandLandmarker initialized!")

# ==========================================
# 3. INITIALIZE WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

sequence = deque(maxlen=20)

prediction = "Waiting..."
confidence = 0.0

frame_count = 0

# Hand landmark connections
connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ==========================================
# 4. REAL-TIME RECOGNITION
# ==========================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("Cannot access webcam!")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Timestamp must increase for VIDEO mode
    timestamp_ms = frame_count * 33
    frame_count += 1

    results = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if results.hand_landmarks:

        hand_landmarks = results.hand_landmarks[0]

        # Draw hand connections
        for start, end in connections:

            x1 = int(hand_landmarks[start].x * frame.shape[1])
            y1 = int(hand_landmarks[start].y * frame.shape[0])

            x2 = int(hand_landmarks[end].x * frame.shape[1])
            y2 = int(hand_landmarks[end].y * frame.shape[0])

            cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # Draw landmark points
        for lm in hand_landmarks:

            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])

            cv2.circle(frame, (x,y), 4, (0,0,255), -1)

        # Extract 21 landmarks (x,y,z)
        landmarks = []

        for lm in hand_landmarks:
            landmarks.extend([lm.x, lm.y, lm.z])

        sequence.append(landmarks)

        # Wait for 20 frames
        if len(sequence) == 20:

            features = np.array(sequence).flatten()

            features = features.reshape(1, -1)

            if features.shape[1] != model.n_features_in_:
                raise ValueError(
                    f"Expected {model.n_features_in_} features, "
                    f"but received {features.shape[1]}. "
                    "Check your dataset structure."
                )

            probabilities = model.predict_proba(features)[0]

            best_index = np.argmax(probabilities)

            prediction = model.classes_[best_index]

            confidence = probabilities[best_index] * 100

    else:
        sequence.clear()
        prediction = "No Hand"
        confidence = 0.0

    # ======================================
    # 5. DISPLAY RESULT
    # ======================================

    cv2.rectangle(
        frame, (10,10), (380,120), (0,0,0), -1
    )

    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (20,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (20,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Frames: {len(sequence)}/20",
        (20,115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        1
    )

    cv2.imshow("ASL Recognition - Random Forest", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# 6. RELEASE RESOURCES
# ==========================================

cap.release()
detector.close()
cv2.destroyAllWindows()

print("Testing finished!")
