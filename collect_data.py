
import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuration
MODEL_PATH = "hand_landmarker.task"
DATASET_PATH = "dataset/asl_sequences.csv"

SEQUENCE_LENGTH = 20

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

# Load existing dataset
if os.path.exists(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH)
else:
    df = pd.DataFrame()

# Initialize MediaPipe
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)

# Open webcam
cap = cv2.VideoCapture(0)

print("ASL Dataset Collection Started")
print("Press A-Z to collect a gesture.")
print("Press Q to quit.")

try:
    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        cv2.putText(
            frame,
            "Press A-Z to collect | Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.imshow("ASL Dataset Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # Check whether a valid letter was pressed
        if chr(key).upper() in LABELS:

            label = chr(key).upper()

            print(f"\nCollecting letter {label}...")

            sequence = []
            frame_count = 0

            while frame_count < SEQUENCE_LENGTH:

                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                result = landmarker.detect(mp_image)

                if result.hand_landmarks:

                    landmarks = result.hand_landmarks[0]

                    frame_data = []

                    for landmark in landmarks:
                        frame_data.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ])

                    sequence.extend(frame_data)

                    frame_count += 1

                    cv2.putText(
                        frame,
                        f"Recording {label}: {frame_count}/20",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                else:
                    cv2.putText(
                        frame,
                        "Hand not detected!",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                cv2.imshow("ASL Dataset Collection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                   print("Recording cancelled.")
                   break


            if frame_count == SEQUENCE_LENGTH:

                row = [label] + sequence

                columns = (
                    ["label"] +
                    [f"feature_{i}" for i in range(1260)]
                )

                new_data = pd.DataFrame(
                    [row],
                    columns=columns
                )

                df = pd.concat(
                    [df, new_data],
                    ignore_index=True
                )

                df.to_csv(DATASET_PATH, index=False)

                print(
                    f"Saved letter {label}. "
                    f"Total samples: {len(df)}"
                )

            else:
                print("Incomplete sequence. Not saved.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

print("Dataset collection ended.")
