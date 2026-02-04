import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import Sequence

# =================================================
# PARAMETERS
# =================================================
IMG_SIZE = 224
SEQ_LEN = 25
STRIDE = 5
BATCH_SIZE = 8
EPOCHS = 2

FRAME_DIR = "extracted_frames_all/train"
WEIGHTS_PATH = "weights/mobilenet_v2_no_top.h5"
MODEL_NAME = "mobilenet_lstm_generator.h5"

# =================================================
# DATA GENERATOR
# =================================================
class VideoSequenceGenerator(Sequence):
    def __init__(self, frame_dir, batch_size=BATCH_SIZE):
        self.samples = []
        self.batch_size = batch_size

        label_map = {"normal": 0, "suspicious": 1}

        for cls, label in label_map.items():
            cls_path = os.path.join(frame_dir, cls)

            for video_folder in os.listdir(cls_path):
                video_path = os.path.join(cls_path, video_folder)
                frames = sorted(os.listdir(video_path))
                total = len(frames)

                for start in range(0, total - SEQ_LEN + 1, STRIDE):
                    self.samples.append((video_path, frames, start, label))

        print("Total training sequences:", len(self.samples))

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, idx):
        batch_samples = self.samples[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        X, y = [], []

        for video_path, frames, start, label in batch_samples:
            seq = []
            for i in range(start, start + SEQ_LEN):
                img = cv2.imread(os.path.join(video_path, frames[i]))
                img = img.astype("float32") / 255.0
                seq.append(img)

            X.append(seq)
            y.append(label)

        return np.array(X), np.array(y)

# =================================================
# CREATE GENERATOR
# =================================================
train_gen = VideoSequenceGenerator(FRAME_DIR)

# =================================================
# LOAD MANUAL MOBILENETV2
# =================================================
base_model = MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.load_weights(WEIGHTS_PATH)
base_model.trainable = False

# =================================================
# BUILD MODEL
# =================================================
model = Sequential([
    TimeDistributed(base_model, input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)),
    TimeDistributed(GlobalAveragePooling2D()),
    LSTM(128),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =================================================
# TRAIN (GENERATOR BASED)
# =================================================
model.fit(
    train_gen,
    epochs=EPOCHS,
    workers=2,
    use_multiprocessing=False
)

# =================================================
# SAVE MODEL
# =================================================
model.save(MODEL_NAME)
print(f"\n🎉 Model saved as {MODEL_NAME}")



import matplotlib.pyplot as plt

# Final accuracy from training output
final_accuracy = 0.9760

plt.figure()
plt.bar(['MobileNet + LSTM'], [final_accuracy])
plt.ylim(0, 1)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Final Training Accuracy')

plt.show()



# Accuracy & loss from training history
accuracy = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(accuracy) + 1)

plt.figure()
plt.plot(epochs, accuracy, label='Training Accuracy')
plt.plot(epochs, loss, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Learning Curve')
plt.legend()

plt.show()

