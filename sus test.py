import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# ==============================
# PARAMETERS (LOCKED)
# ==============================
IMG_SIZE = 224
SEQ_LEN = 25
THRESHOLD = 0.60   # ✅ FINAL THRESHOLD
MODEL_PATH = "mobilenet_lstm_generator.h5"

LABELS = {0: "NORMAL", 1: "SUSPICIOUS"}

# ==============================
# LOAD MODEL
# ==============================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ==============================
# FRAME BUFFER
# ==============================
frame_buffer = deque(maxlen=SEQ_LEN)
confidence_buffer = deque(maxlen=5)  # 🔥 smoothing

# ==============================
# WEBCAM
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not detected")
    exit()

print("🎥 Webcam started (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # --------------------------
    # PREPROCESS FRAME
    # --------------------------
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    frame_buffer.append(img)

    text = "Collecting frames..."
    color = (255, 255, 0)

    # --------------------------
    # PREDICTION
    # --------------------------
    if len(frame_buffer) == SEQ_LEN:
        seq = np.expand_dims(frame_buffer, axis=0)
        conf = model.predict(seq, verbose=0)[0][0]

        confidence_buffer.append(conf)
        avg_conf = np.mean(confidence_buffer)

        if avg_conf >= THRESHOLD:
            label = 1
            color = (0, 0, 255)
        else:
            label = 0
            color = (0, 255, 0)

        text = f"{LABELS[label]} | Conf: {avg_conf:.3f}"

        # -------- PRINT TO SHELL --------
        print(f"PRED: {LABELS[label]} | CONFIDENCE: {avg_conf:.3f}")

    # --------------------------
    # DISPLAY
    # --------------------------
    cv2.putText(
        display,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Live Suspicious Activity Detection", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam stopped")
