
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_DIR = "dataset/word_sequences"
LABEL_FILE = "dataset/word_labels.csv"
MODEL_PATH = "hand_landmarker.task"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# ==========================================
# 2. INITIALISE MEDIAPIPE TASKS
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
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

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# ==========================================
# 3. LOAD EXISTING LABELS
# ==========================================

if os.path.exists(LABEL_FILE):
    df = pd.read_csv(LABEL_FILE)
else:
    df = pd.DataFrame(columns=["filename", "text"])

# ==========================================
# 4. START CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    detector.close()
    exit()

recording = False
sequence = []

word = input("Enter word to collect: ").strip().upper()

if not word:
    print("Error: Word cannot be empty.")
    cap.release()
    detector.close()
    exit()

print("\nControls:")
print("R = Start recording")
print("S = Save sequence")
print("Q = Quit")

# MediaPipe VIDEO mode requires increasing timestamps
previous_timestamp = 0

# ==========================================
# 5. MAIN CAMERA LOOP
# ==========================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("Cannot receive camera frame.")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert frame into MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Generate increasing timestamp
    timestamp_ms = time.monotonic_ns() // 1_000_000
    timestamp_ms = max(timestamp_ms, previous_timestamp + 1)
    previous_timestamp = timestamp_ms

    # Detect hand landmarks
    results = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    landmarks = np.zeros(63, dtype=np.float32)

    hand_detected = False

    if results.hand_landmarks:

        hand = results.hand_landmarks[0]

        hand_detected = True

        # Extract 21 landmarks x 3 coordinates
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand],
            dtype=np.float32
        ).flatten()

        # Draw hand connections
        h, w, _ = frame.shape

        points = []

        for lm in hand:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 0, 0),
                2
            )

    # Record detected hand frames
    if recording and hand_detected:
        sequence.append(landmarks.copy())

    # ======================================
    # 6. DISPLAY INFORMATION
    # ======================================

    status = "RECORDING" if recording else "READY"

    cv2.putText(
        frame,
        f"Word: {word} | {status}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if recording else (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frames: {len(sequence)}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "R: Record | S: Save | Q: Quit",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("ASL Word Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # ======================================
    # 7. KEYBOARD CONTROLS
    # ======================================

    if key == ord('r'):

        sequence = []
        recording = True

        print("Recording started...")

    elif key == ord('s'):

        if recording and len(sequence) > 0:

            recording = False

            word_sequence = np.array(
                sequence,
                dtype=np.float32
            )

            filename = f"{word}_{len(df)+1:04d}.npy"

            save_path = os.path.join(
                DATA_DIR,
                filename
            )

            np.save(save_path, word_sequence)

            new_row = pd.DataFrame([{
                "filename": filename,
                "text": word
            }])

            df = pd.concat(
                [df, new_row],
                ignore_index=True
            )

            df.to_csv(LABEL_FILE, index=False)

            print("\nSequence saved successfully!")
            print("Filename:", filename)
            print("Word:", word)
            print("Shape:", word_sequence.shape)

        else:
            print("No sequence to save.")

    elif key == ord('q'):
        break

# ==========================================
# 8. RELEASE RESOURCES
# ==========================================

cap.release()
cv2.destroyAllWindows()
detector.close()

print("Word collection completed.")
