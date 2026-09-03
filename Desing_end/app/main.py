"""
main.py — FastAPI backend for BananaVision
"""

import os
import io
import base64
import time
import tempfile
import traceback
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

import pipeline as pl

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODELS_DIR  = BASE_DIR / "models"
STATIC_DIR  = BASE_DIR / "static"

SEG_MODEL_PATH      = MODELS_DIR / "banana_maskrcnn_finetuned.pth"
RIPENESS_MODEL_PATH = MODELS_DIR / "banana_mobilenetv2_best.pth"


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup and store them in app.state."""
    print("[startup] Loading models …")
    try:
        app.state.device = pl.DEVICE
        app.state.seg_model      = pl.load_segmentation_model(
            str(SEG_MODEL_PATH), pl.DEVICE
        )
        app.state.ripeness_model = pl.load_ripeness_model(
            str(RIPENESS_MODEL_PATH), pl.DEVICE
        )
        app.state.models_loaded = True
        print(f"[startup] Models loaded on {pl.DEVICE}")
    except Exception as exc:
        print(f"[startup] WARNING — could not load models: {exc}")
        app.state.seg_model      = None
        app.state.ripeness_model = None
        app.state.models_loaded  = False
    yield
    # shutdown: nothing to clean up


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI-Based Fruit Quality Detection API (BananaVision)",
    description="Fruit Quality & Ripeness Detection system, currently specialized and fine-tuned for Bananas.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Utility ───────────────────────────────────────────────────────────────────
def _img_to_base64(img_rgb: np.ndarray) -> str:
    """Encode an RGB numpy array as a base64 PNG string."""
    _, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _pil_to_base64(pil_img: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index_path))


@app.get("/health")
async def health():
    return {
        "status"      : "ok",
        "device"      : str(app.state.device),
        "models_loaded": app.state.models_loaded,
    }


@app.get("/samples")
async def list_samples():
    samples_dir = STATIC_DIR / "samples"
    if not samples_dir.exists():
        return []
    items = []
    sample_meta = {
        "sample_unripe.jpg": {"title": "Unripe (Green)", "stage": "unripe", "desc": "Firm green banana"},
        "sample_ripe.jpg": {"title": "Perfect Ripe", "stage": "ripe", "desc": "Bright golden yellow"},
        "sample_overripe.jpg": {"title": "Sweet Overripe", "stage": "overripe", "desc": "Spotted sugar-rich"},
        "sample_bunch.jpeg": {"title": "Fresh Banana Bunch", "stage": "mixed", "desc": "Multi-banana cluster"},
    }
    for file in samples_dir.iterdir():
        if file.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            meta = sample_meta.get(file.name, {"title": file.stem.replace("_", " ").title(), "stage": "banana", "desc": "Test sample"})
            items.append({
                "filename": file.name,
                "url": f"/static/samples/{file.name}",
                **meta
            })
    return items


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accepts a JPEG or PNG image, runs the two-stage pipeline,
    and returns JSON with annotated image + per-banana results.
    """
    content_type = file.content_type or ""
    if content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {content_type}. Use JPEG or PNG.",
        )

    # ── Save to a temp file (pipeline.py uses cv2.imread) ────────────────────
    suffix = ".jpg" if "jpeg" in content_type or "jpg" in content_type else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()

        if app.state.models_loaded:
            output = pl.run_banana_pipeline(
                tmp_path,
                app.state.seg_model,
                app.state.ripeness_model,
                app.state.device,
            )
            mode_tag = "Deep Learning (Mask R-CNN + MobileNetV2)"
        else:
            output = pl.run_cv_fallback_pipeline(tmp_path)
            mode_tag = "Computer Vision (HSV Color Analysis)"

        elapsed_ms = (time.perf_counter() - t0) * 1000

        image_np     = output["image_np"]
        num_detected = output["num_detected"]
        results      = output["results"]

        # ── Annotated image ───────────────────────────────────────────────────
        annotated_np  = pl.generate_annotated_image(image_np, results)
        annotated_b64 = _img_to_base64(annotated_np)

        # ── Build JSON response ───────────────────────────────────────────────
        summary = Counter(r["ripeness"] for r in results)
        summary_dict = {
            "ripe"    : summary.get("ripe",     0),
            "unripe"  : summary.get("unripe",   0),
            "overripe": summary.get("overripe", 0),
            "rotten"  : summary.get("rotten",   0),
        }

        results_json = []
        for r in results:
            probs_arr = r["all_probs"]  # numpy [4], order: overripe, ripe, rotten, unripe
            all_probs_dict = {
                pl.IDX_TO_CLASS[i]: float(probs_arr[i]) for i in range(4)
            }
            results_json.append({
                "banana_id"   : r["banana_id"],
                "ripeness"    : r["ripeness"],
                "confidence"  : float(r["confidence"]),
                "all_probs"   : all_probs_dict,
                "seg_score"   : float(r["seg_score"]),
                "bbox"        : [int(v) for v in r["bbox"]],
                "low_conf"    : bool(r["low_conf"]),
                "exposure_tag": r["exposure_tag"],
                "crop_image"  : _pil_to_base64(r["crop_pil"]),
                "insight"     : r.get("insight", {}),
            })

        return JSONResponse({
            "num_detected"      : num_detected,
            "annotated_image"   : annotated_b64,
            "results"           : results_json,
            "summary"           : summary_dict,
            "processing_time_ms": round(elapsed_ms, 1),
            "mode"              : mode_tag,
        })

    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
