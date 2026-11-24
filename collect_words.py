import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_DIR = "dataset/word_sequences"
LABEL_FILE = "dataset/word_labels.csv"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("dataset", exist_ok=True)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==========================================
# 2. LOAD EXISTING LABELS
# ==========================================

if os.path.exists(LABEL_FILE):
    df = pd.read_csv(LABEL_FILE)
else:
    df = pd.DataFrame(columns=["filename", "text"])

# ==========================================
# 3. START CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

recording = False
sequence = []

word = input("Enter word to collect: ").strip().upper()

print("\nControls:")
print("R = Start recording")
print("S = Save sequence")
print("Q = Quit")

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmarks = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        for lm in hand.landmark:
            # Same 63-feature representation:
            # x, y, z for each of 21 landmarks
            pass

        landmarks = np.array(
            [[lm.x, lm.y, lm.z]
             for lm in hand.landmark],
            dtype=np.float32
        ).flatten()

        mp_draw.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS
        )

    if recording and results.multi_hand_landmarks:
    sequence.append(landmarks.copy())


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

    cv2.imshow("ASL Word Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        sequence = []
        recording = True
        print("Recording started...")

    elif key == ord('s'):

        if recording and len(sequence) > 0:

            recording = False

            word_sequence = np.array(
                sequence, dtype=np.float32
            )

            filename = f"{word}_{len(df)+1:04d}.npy"

            save_path = os.path.join(
                DATA_DIR, filename
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

            print("Saved:", filename)
            print("Shape:", word_sequence.shape)

        else:
            print("No sequence to save.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

