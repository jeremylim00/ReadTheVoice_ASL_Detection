
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = "models/asl_lstm_ctc.keras"
HAND_MODEL_PATH = "hand_landmarker.task"

FEATURES = 63

# Gesture segmentation settings
MOTION_START_THRESHOLD = 0.003
MOTION_END_THRESHOLD = 0.0015

START_CONFIRM_FRAMES = 2
IDLE_SECONDS = 0.8
MIN_SEQUENCE_FRAMES = 8

# Display settings
PREDICTION_DISPLAY_SECONDS = 2.5

# Webcam settings
CAMERA_INDEX = 0


# ==========================================
# 2. LOAD LSTM MODEL
# ==========================================

print("Loading ASL LSTM model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# 3. CHARACTER MAPPING
# ==========================================

NUM_TO_CHAR = {
    i + 1: chr(i + 65)
    for i in range(26)
}


# ==========================================
# 4. INITIALISE MEDIAPIPE
# ==========================================

base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# ==========================================
# 5. CTC PREDICTION
# ==========================================

def predict_word(sequence):

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if len(sequence) < MIN_SEQUENCE_FRAMES:
        return ""

    input_data = np.expand_dims(
        sequence,
        axis=0
    )

    logits = model(
        input_data,
        training=False
    )

    decoded, log_prob = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, [1, 0, 2]),
        sequence_length=tf.constant(
            [len(sequence)],
            dtype=tf.int32
        ),
        merge_repeated=True,
        blank_index=0
    )

    result = tf.sparse.to_dense(
        decoded[0],
        default_value=-1
    ).numpy()[0]

    predicted_text = "".join(
        NUM_TO_CHAR[i]
        for i in result
        if i in NUM_TO_CHAR
    )

    return predicted_text


# ==========================================
# 6. EXTRACT HAND LANDMARKS
# ==========================================

def extract_landmarks(results, frame):

    if not results.hand_landmarks:
        return None

    hand = results.hand_landmarks[0]

    landmarks = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand],
        dtype=np.float32
    )

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

    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]

    for start, end in connections:

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 0, 0),
            2
        )

    return landmarks.flatten()


# ==========================================
# 7. CALCULATE HAND MOVEMENT
# ==========================================

def calculate_motion(current, previous):

    if previous is None:
        return 0.0

    current = current.reshape(21, 3)
    previous = previous.reshape(21, 3)

    distances = np.linalg.norm(
        current - previous,
        axis=1
    )

    return float(np.mean(distances))


# ==========================================
# 8. INITIALISE VARIABLES
# ==========================================

recording = False

sequence = []

previous_landmarks = None

motion_hits = 0

last_motion_time = 0

last_prediction = ""

prediction_time = 0

pre_buffer = deque(maxlen=5)

previous_timestamp = 0


# ==========================================
# 9. START WEBCAM
# ==========================================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("Cannot open webcam.")

    detector.close()

    raise SystemExit


print("\nASL Live Word Recognition Started!")
print("Press Q to quit.")

# ==========================================
# 10. MAIN WEBCAM LOOP
# ==========================================

try:

    while cap.isOpened():

        # ----------------------------------
        # Read webcam
        # ----------------------------------

        ret, frame = cap.read()

        if not ret:

            print("Cannot read webcam frame.")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = time.monotonic_ns() // 1_000_000

        timestamp_ms = max(
            timestamp_ms,
            previous_timestamp + 1
        )

        previous_timestamp = timestamp_ms

        results = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        current_time = time.time()

        # ----------------------------------
        # Extract hand landmarks
        # ----------------------------------

        landmarks = extract_landmarks(
            results,
            frame
        )

        hand_detected = landmarks is not None

        motion = 0.0

        # ----------------------------------
        # SECTION 10A: GESTURE DETECTION
        # ----------------------------------

        if hand_detected:

            motion = calculate_motion(
                landmarks,
                previous_landmarks
            )

            if not recording:

                # Store recent frames before signing
                pre_buffer.append(
                    landmarks.copy()
                )

                # Confirm movement before recording
                if motion > MOTION_START_THRESHOLD:

                    motion_hits += 1

                else:

                    motion_hits = 0

                # Start recording
                if motion_hits >= START_CONFIRM_FRAMES:

                    recording = True

                    sequence = list(pre_buffer)

                    last_motion_time = current_time

                    motion_hits = 0

                    print("Gesture recording started...")

            else:

                # Continue recording gesture
                sequence.append(
                    landmarks.copy()
                )

                # Update last movement time
                if motion > MOTION_END_THRESHOLD:

                    last_motion_time = current_time

            previous_landmarks = landmarks.copy()

        else:

            previous_landmarks = None

            if not recording:

                motion_hits = 0

        # ----------------------------------
        # SECTION 10B: END GESTURE RECORDING
        # ----------------------------------

        if recording:

            idle_time = (
                current_time - last_motion_time
            )

            if idle_time >= IDLE_SECONDS:

                print(
                    "Gesture completed. Frames:",
                    len(sequence)
                )

                recording = False

                # ----------------------------------
                # SECTION 11: LSTM + CTC PREDICTION
                # ----------------------------------

                if len(sequence) >= MIN_SEQUENCE_FRAMES:

                    try:

                        predicted_word = predict_word(
                            sequence
                        )

                        if predicted_word:

                            last_prediction = predicted_word

                            prediction_time = current_time

                            print(
                                "Predicted word:",
                                predicted_word
                            )

                        else:

                            print("No word recognized.")

                    except Exception as e:

                        print("Prediction error:", e)

                else:

                    print("Sequence too short.")

                # Reset gesture variables
                sequence = []

                pre_buffer.clear()

                motion_hits = 0

        # ----------------------------------
        # SECTION 12: DISPLAY INFORMATION
        # ----------------------------------

        if recording:

            status = "RECORDING"

            status_color = (0, 0, 255)

        else:

            status = "WAITING FOR GESTURE"

            status_color = (0, 255, 255)

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        # Display hand detection status
        if hand_detected:

            hand_status = "Hand Detected"

        else:

            hand_status = "No Hand Detected"

        cv2.putText(
            frame,
            hand_status,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Display movement value
        cv2.putText(
            frame,
            f"Motion: {motion:.5f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Display recording frame count
        if recording:

            cv2.putText(
                frame,
                f"Frames: {len(sequence)}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # Display latest prediction
        if (
            last_prediction
            and current_time - prediction_time
            < PREDICTION_DISPLAY_SECONDS
        ):

            cv2.putText(
                frame,
                "Prediction: " + last_prediction,
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

        cv2.putText(
            frame,
            "Press Q to Quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ASL Live Word Recognition",
            frame
        )

        # ----------------------------------
        # Q TO QUIT
        # ----------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):

            print("Quitting ASL recognition...")
            break


# ==========================================
# 13. RELEASE RESOURCES
# ==========================================

finally:

    cap.release()

    cv2.destroyAllWindows()

    detector.close()

    print("ASL Live Recognition Closed.")
