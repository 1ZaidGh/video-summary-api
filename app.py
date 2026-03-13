import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
import shutil
import os

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

app = FastAPI()

MODEL_PATH = "model.h5"

# -------------------------
# CNN Feature extractor
# -------------------------
cnn = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# project 1280 → 512
feature_projection = tf.keras.layers.Dense(512)

# -------------------------
# Caption model
# -------------------------
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

# -------------------------
# Load summarization model
# -------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


# -------------------------
# Extract frames
# -------------------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()
    return np.array(frames), fps


# -------------------------
# Extract CNN features
# -------------------------
def extract_features(frames):
    frames_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(
        frames.astype(np.float32)
    )
    features = cnn.predict(frames_preprocessed, verbose=0)
    features = feature_projection(features)
    return features.numpy()


# -------------------------
# Caption a frame
# -------------------------
def caption_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    inputs = processor(images=image, return_tensors="pt")
    out = caption_model.generate(**inputs, max_new_tokens=30)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# -------------------------
# Get all scenes with timestamps
# -------------------------
def get_all_scenes(frames, fps, step=100):
    captions = []
    seen = set()
    indices = range(0, len(frames), step)

    for idx in indices:
        caption = caption_frame(frames[idx])

        # Skip duplicate captions
        if caption not in seen:
            seen.add(caption)
            timestamp = idx / fps if fps > 0 else idx / 30
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            captions.append({
                "timestamp": f"{minutes:02d}:{seconds:02d}",
                "frame": int(idx),
                "caption": caption
            })

    return captions


# -------------------------
# API endpoint - full scene list
# -------------------------
@app.post("/summarize")
async def summarize_video(file: UploadFile = File(...)):
    path = f"temp_{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        frames, fps = extract_frames(path)

        if len(frames) == 0:
            return {"error": "Could not extract frames from video."}

        print(f"✅ Extracted {len(frames)} frames at {fps:.1f} fps")

        # Get all scenes (every ~3 seconds by default)
        scenes = get_all_scenes(frames, fps, step=100)

        # Also build a plain text summary
        summary = ". ".join([s["caption"] for s in scenes])
        if not summary.endswith("."):
            summary += "."

        return {
            "total_frames": len(frames),
            "fps": round(fps, 2),
            "scenes_found": len(scenes),
            "scenes": scenes,
            "summary": summary
        }

    finally:
        if os.path.exists(path):
            os.remove(path)


# -------------------------
# Home endpoint
# -------------------------
@app.get("/")
def home():
    return {"message": "Video summarization API running"}


# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)