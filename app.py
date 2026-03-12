import cv2
import h5py
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
import shutil
import os

app = FastAPI()

MODEL_PATH = "model.h5"
DATASET_PATH = "SumMe.h5"

MAX_LEN = 500


# -------------------------
# Pad sequences for training
# -------------------------
def pad_sequences(features, labels):

    padded_X = []
    padded_Y = []

    for f, l in zip(features, labels):

        length = min(len(f), MAX_LEN)

        x = f[:length]
        y = l[:length]

        if length < MAX_LEN:

            pad_x = np.zeros((MAX_LEN - length, f.shape[1]))
            pad_y = np.zeros((MAX_LEN - length,))

            x = np.vstack([x, pad_x])
            y = np.concatenate([y, pad_y])

        padded_X.append(x)
        padded_Y.append(y)

    return np.array(padded_X), np.array(padded_Y)


# -------------------------
# Load dataset
# -------------------------
def load_dataset():

    dataset = []

    with h5py.File(DATASET_PATH, "r") as f:

        for key in f.keys():

            video = f[key]

            features = np.array(video["feature"])
            scores = np.array(video["label"])

            dataset.append((features, scores))

    return dataset


# -------------------------
# Build model
# -------------------------
def build_model(input_dim):

    inputs = tf.keras.Input(shape=(None, input_dim))

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True)
    )(inputs)

    x = tf.keras.layers.Dense(64, activation="relu")(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model


# -------------------------
# Train model
# -------------------------
def train_model():

    dataset = load_dataset()

    features = []
    scores = []

    for f, s in dataset:
        features.append(f)
        scores.append(s)

    X, Y = pad_sequences(features, scores)

    input_dim = X.shape[2]

    model = build_model(input_dim)

    model.fit(
        X,
        Y,
        epochs=5,
        batch_size=2
    )

    model.save(MODEL_PATH)

    return model


# -------------------------
# Load model
# -------------------------
if os.path.exists(MODEL_PATH):

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

else:

    print("Training model...")
    model = train_model()


# -------------------------
# Extract frames
# -------------------------
def extract_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (224,224))

        frames.append(frame)

    cap.release()

    return np.array(frames)


# -------------------------
# Convert scores to text
# -------------------------
def scores_to_text(scores):

    threshold = 0.5
    important_frames = np.where(scores > threshold)[0]

    if len(important_frames) == 0:
        return "No important events were detected in the video."

    segments = []
    start = important_frames[0]

    for i in range(1, len(important_frames)):

        if important_frames[i] != important_frames[i-1] + 1:
            segments.append((start, important_frames[i-1]))
            start = important_frames[i]

    segments.append((start, important_frames[-1]))

    sentences = []

    for i, (s, e) in enumerate(segments):

        sentences.append(
            f"Event {i+1} occurs between frame {s} and {e}."
        )

    return " ".join(sentences)


# -------------------------
# API endpoint
# -------------------------
@app.post("/summarize")

async def summarize_video(file: UploadFile = File(...)):

    path = f"temp_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    frames = extract_frames(path)

    frames = frames / 255.0

    frames = np.expand_dims(frames, axis=0)

    scores = model.predict(frames)[0].flatten()

    summary_text = scores_to_text(scores)

    return {
        "summary": summary_text
    }

    if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)