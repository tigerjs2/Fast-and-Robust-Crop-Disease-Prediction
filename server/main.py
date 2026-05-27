import asyncio
import base64
import io
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from base_model.SAM import SAM2MaskGenerator
from scripts.SecondStageTest import build_model_from_checkpoint, load_checkpoint
from scripts.SecondStageTrain import build_image_transform, load_crop_text_embeddings


SAM_WEIGHTS = Path(os.environ.get("SAM_WEIGHTS", PROJECT_ROOT / "weights" / "sam2.1_t.pt"))
CLASSIFIER_WEIGHTS = Path(
    os.environ.get("CLASSIFIER_WEIGHTS", PROJECT_ROOT / "weights" / "best_classifier.pth")
)
TEXT_EMBEDDING_DIR = Path(
    os.environ.get("TEXT_EMBEDDING_DIR", PROJECT_ROOT / "data" / "TextEmbeddings")
)


STATE: dict = {}
INFER_LOCK = asyncio.Lock()


def _discover_crops(text_embedding_dir: Path) -> list[str]:
    if not text_embedding_dir.exists():
        return []
    return sorted({p.stem.lower() for p in text_embedding_dir.glob("*.pt")})


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(CLASSIFIER_WEIGHTS, device)
    class_names = checkpoint.get("class_names", [])
    if not class_names:
        raise RuntimeError("Checkpoint missing class_names")

    model = build_model_from_checkpoint(checkpoint, num_classes=len(class_names), device=device)
    model.eval()

    config = checkpoint.get("config", {})
    image_size = int(config.get("image_size", 224))
    transform = build_image_transform(image_size=image_size, apply_augmentation=False)

    crops = _discover_crops(TEXT_EMBEDDING_DIR)
    text_embeddings = (
        load_crop_text_embeddings(
            text_embedding_dir=TEXT_EMBEDDING_DIR,
            crop_names=crops,
            device=device,
            expected_num_queries=5,
        )
        if crops
        else {}
    )

    sam_generator = SAM2MaskGenerator(model_path=str(SAM_WEIGHTS), mode="box")
    try:
        sam_generator.model.to(str(device))
    except Exception:
        pass

    STATE.update(
        device=device,
        model=model,
        class_names=list(class_names),
        transform=transform,
        text_embeddings=text_embeddings,
        sam_generator=sam_generator,
    )
    yield
    STATE.clear()


app = FastAPI(title="Crop Disease Inference Server", lifespan=lifespan)


class PredictResponse(BaseModel):
    pred_index: int
    pred_class: str
    pred_confidence: float
    probs: list[float]
    class_names: list[str]
    region_info: dict[str, float]
    masked_image_png_base64: Optional[str] = None


@app.get("/health")
def health():
    device = STATE.get("device")
    return {"status": "ok", "device": str(device) if device is not None else "uninitialized"}


@app.get("/crops")
def crops():
    return {"crops": sorted(STATE.get("text_embeddings", {}).keys())}


@app.get("/classes")
def classes():
    return {"class_names": STATE.get("class_names", [])}


def _parse_bbox(bbox_str: str) -> Tuple[int, int, int, int]:
    parts = [p for p in bbox_str.replace(" ", "").split(",") if p]
    if len(parts) != 4:
        raise HTTPException(status_code=422, detail="bbox must be 'x1,y1,x2,y2'")
    try:
        coords = [int(float(p)) for p in parts]
    except ValueError:
        raise HTTPException(status_code=422, detail="bbox values must be numeric")
    return coords[0], coords[1], coords[2], coords[3]


def _run_inference_sync(image_path: str, crop: str, bbox: Tuple[int, int, int, int],
                        return_masked_image: bool) -> dict:
    device = STATE["device"]
    text_embeddings = STATE["text_embeddings"]
    if crop not in text_embeddings:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown crop '{crop}'. Available: {sorted(text_embeddings.keys())}",
        )

    sam = STATE["sam_generator"]
    masked_bgr, region_info, _ = sam.generate_with_region(image_path=image_path, bbox=list(bbox))

    masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = STATE["transform"](masked_rgb).unsqueeze(0).to(device)
    text_queries = text_embeddings[crop].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = STATE["model"](image_tensor, text_queries)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        pred_conf = float(probs[0, pred_idx].item())
        probs_list = probs.squeeze(0).detach().cpu().tolist()

    class_names = STATE["class_names"]
    result = {
        "pred_index": pred_idx,
        "pred_class": class_names[pred_idx],
        "pred_confidence": pred_conf,
        "probs": probs_list,
        "class_names": class_names,
        "region_info": region_info,
        "masked_image_png_base64": None,
    }

    if return_masked_image:
        ok, buf = cv2.imencode(".png", masked_bgr)
        if ok:
            result["masked_image_png_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")

    return result


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    crop: str = Form(...),
    bbox: str = Form(..., description="x1,y1,x2,y2"),
    return_masked_image: bool = Form(False),
):
    bbox_xyxy = _parse_bbox(bbox)
    crop_norm = crop.strip().lower()

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload")

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(raw)
        tmp.flush()
        tmp.close()

        async with INFER_LOCK:
            result = await asyncio.to_thread(
                _run_inference_sync, tmp.name, crop_norm, bbox_xyxy, return_masked_image
            )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return result
