import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_DIR = "dataset/word_sequences"
LABEL_FILE = "dataset/word_labels.csv"

FEATURES = 63
NUM_CLASSES = 27  # Blank + A-Z

EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# ==========================================
# 2. CHARACTER ENCODING
# ==========================================

CHAR_TO_NUM = {
    chr(i + 65): i + 1 for i in range(26)
}

NUM_TO_CHAR = {
    v: k for k, v in CHAR_TO_NUM.items()
}

def encode_text(text):

    return [
        CHAR_TO_NUM[c]
        for c in text.upper()
        if c in CHAR_TO_NUM
    ]

# ==========================================
# 3. LOAD DATASET
# ==========================================

print("Loading word sequences...")

df = pd.read_csv(LABEL_FILE)

X = []
Y = []

for _, row in df.iterrows():

    path = os.path.join(
        DATA_DIR, row["filename"]
    )

    sequence = np.load(path)

    if sequence.ndim != 2 or sequence.shape[1] != FEATURES:
        print("Skipping invalid sequence:", path)
        continue

    label = encode_text(row["text"])

    if len(label) == 0:
        continue

    # CTC needs enough timesteps for repeated characters.
    # A conservative minimum is label length plus
    # the number of adjacent repeated characters.
    min_frames = len(label) + sum(
        label[i] == label[i-1]
        for i in range(1, len(label))
    )

    if len(sequence) < min_frames:
        print("Skipping too-short sequence:", path)
        continue

    X.append(sequence.astype(np.float32))
    Y.append(label)

print("Total sequences:", len(X))

# ==========================================
# 4. TRAIN / VALIDATION SPLIT
# ==========================================

indices = np.arange(len(X))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42
)

# ==========================================
# 5. BUILD LSTM MODEL
# ==========================================

inputs = tf.keras.Input(
    shape=(None, FEATURES),
    name="landmarks"
)

x = tf.keras.layers.Masking(
    mask_value=0.0
)(inputs)

x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(
        128,
        return_sequences=True
    )
)(x)

x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(
        64,
        return_sequences=True
    )
)(x)

outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    name="logits"
)(x)

model = tf.keras.Model(inputs, outputs)

model.summary()

# ==========================================
# 6. PAD BATCHES
# ==========================================

def make_batch(indices):

    batch_x = [X[i] for i in indices]
    batch_y = [Y[i] for i in indices]

    input_lengths = np.array(
        [len(s) for s in batch_x],
        dtype=np.int32
    )

    label_lengths = np.array(
        [len(s) for s in batch_y],
        dtype=np.int32
    )

    max_time = max(input_lengths)
    max_label = max(label_lengths)

    padded_x = np.zeros(
        (len(batch_x), max_time, FEATURES),
        dtype=np.float32
    )

    padded_y = np.zeros(
        (len(batch_y), max_label),
        dtype=np.int32
    )

    for i, (seq, label) in enumerate(
        zip(batch_x, batch_y)
    ):

        padded_x[i, :len(seq)] = seq
        padded_y[i, :len(label)] = label

    return (
        padded_x,
        padded_y,
        input_lengths,
        label_lengths
    )

# ==========================================
# 7. CTC TRAINING
# ==========================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

@tf.function
def train_step(batch_x, batch_y, input_len, label_len):

    with tf.GradientTape() as tape:

        logits = model(batch_x, training=True)

        loss = tf.nn.ctc_loss(
            labels=batch_y,
            logits=logits,
            label_length=label_len,
            logit_length=input_len,
            logits_time_major=False,
            blank_index=0
        )

        loss = tf.reduce_mean(loss)

    gradients = tape.gradient(
        loss, model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )

    return loss

@tf.function
def validation_step(batch_x, batch_y, input_len, label_len):

    logits = model(batch_x, training=False)

    loss = tf.nn.ctc_loss(
        labels=batch_y,
        logits=logits,
        label_length=label_len,
        logit_length=input_len,
        logits_time_major=False,
        blank_index=0
    )

    return tf.reduce_mean(loss)

# ==========================================
# 8. TRAINING LOOP
# ==========================================

for epoch in range(EPOCHS):

    np.random.shuffle(train_idx)

    train_losses = []

    for start in range(0, len(train_idx), BATCH_SIZE):

        batch_ids = train_idx[
            start:start + BATCH_SIZE
        ]

        bx, by, il, ll = make_batch(batch_ids)

        loss = train_step(
            bx, by, il, ll
        )

        train_losses.append(float(loss.numpy()))

    val_losses = []

    for start in range(0, len(val_idx), BATCH_SIZE):

        batch_ids = val_idx[
            start:start + BATCH_SIZE
        ]

        bx, by, il, ll = make_batch(batch_ids)

        val_loss = validation_step(
            bx, by, il, ll
        )

        val_losses.append(float(val_loss.numpy()))

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {np.mean(train_losses):.4f} | "
        f"Val Loss: {np.mean(val_losses):.4f}"
    )

# ==========================================
# 9. SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

model.save("models/asl_lstm_ctc.keras")

print("LSTM model saved successfully!")

