import os, io, asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image

MODEL_LOCAL_PATH = "/tmp/model.pt"
MODEL_HF_REPO = os.environ.get("MODEL_HF_REPO")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "model.pt")

app = FastAPI()
app.state.model = None

@app.on_event("startup")
async def load_model():
    if not MODEL_HF_REPO:
        raise RuntimeError("Set MODEL_HF_REPO env var!")
    model_path = await asyncio.to_thread(hf_hub_download, repo_id=MODEL_HF_REPO, filename=MODEL_FILENAME)
    app.state.model = YOLO(model_path)
    print("✅ Model loaded")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = 0.25):
    if app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded yet.")
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    np_img = np.array(img)
    results = app.state.model.predict(source=np_img, conf=conf, imgsz=640, device="cpu")
    annotated = results[0].plot()
    out_img = Image.fromarray(annotated)
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
