import numpy as np
import tensorflow as tf
import sys

# ==========================================
# 1. LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    "models/asl_lstm_ctc.keras"
)

# ==========================================
# 2. CHARACTER MAPPING
# ==========================================

NUM_TO_CHAR = {
    i + 1: chr(i + 65)
    for i in range(26)
}

# ==========================================
# 3. PREDICTION FUNCTION
# ==========================================

def predict_word(sequence):

    sequence = np.asarray(
        sequence, dtype=np.float32
    )

    if sequence.ndim != 2 or sequence.shape[1] != 63:
        raise ValueError("Expected sequence shape (T, 63)")

    input_data = np.expand_dims(sequence, axis=0)

    logits = model(
        input_data, training=False
    )

    # CTC greedy decoding
    decoded, log_prob = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(logits, [1, 0, 2]),
        sequence_length=tf.constant(
            [len(sequence)], dtype=tf.int32
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
# 4. TEST SAVED SEQUENCE
# ==========================================

if len(sys.argv) < 2:
    print("Usage: python predict_sequence.py <file.npy>")
    sys.exit(1)

sequence = np.load(sys.argv[1])

prediction = predict_word(sequence)

print("\n============================")
print("ASL SEQUENCE PREDICTION")
print("============================")
print("Predicted word:", prediction)
print("============================")

