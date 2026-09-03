# 🍌 BananaVision — AI-Powered Ripeness Detection

A production-grade web application that uses a **two-stage deep learning pipeline** (Mask R-CNN → MobileNetV2) to detect individual bananas in an image and classify each one's ripeness in real time.

---

## Architecture

```
Upload Image → Mask R-CNN Segmentation → Crop Each Banana
                                              ↓
                               MobileNetV2 + 5-Transform TTA
                                              ↓
                              Ripeness: unripe / ripe / overripe / rotten
```

| Stage | Model | Output |
|-------|-------|--------|
| Segmentation | Mask R-CNN ResNet-50 FPN | Binary masks, bounding boxes, confidence scores |
| Classification | MobileNetV2 (1280→4) | Ripeness class + confidence + class probabilities |

---

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional but recommended for speed)

---

## Setup

### 1. Clone / place model files

```
Desing_end/
├── app/
│   ├── main.py
│   ├── pipeline.py
│   ├── models/
│   │   ├── banana_maskrcnn_finetuned.pth   ← place here
│   │   └── banana_mobilenetv2_best.pth     ← place here
│   └── static/
│       └── index.html
└── requirements.txt
```

The `.pth` files are already present at the root of your workspace — **copy or move them into `app/models/`**:

```powershell
Copy-Item banana_maskrcnn_finetuned.pth app\models\
Copy-Item banana_mobilenetv2_best.pth   app\models\
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> For GPU support, install the CUDA build of PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 3. Run the server

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the browser

```
http://localhost:8000
```

---

## API Reference

### `GET /health`
Returns server status, device (cuda/cpu), and whether models are loaded.

```json
{ "status": "ok", "device": "cuda", "models_loaded": true }
```

### `POST /analyze`
Upload a JPEG or PNG image. Returns JSON:

```json
{
  "num_detected": 3,
  "annotated_image": "<base64 PNG>",
  "results": [
    {
      "banana_id": 1,
      "ripeness": "ripe",
      "confidence": 0.923,
      "all_probs": { "overripe": 0.02, "ripe": 0.92, "rotten": 0.01, "unripe": 0.04 },
      "seg_score": 0.97,
      "bbox": [120, 80, 340, 420],
      "low_conf": false,
      "exposure_tag": "",
      "crop_image": "<base64 PNG>"
    }
  ],
  "summary": { "ripe": 2, "unripe": 1, "overripe": 0, "rotten": 0 },
  "processing_time_ms": 312.4
}
```

---

## ML Pipeline Details

| Parameter | Value |
|-----------|-------|
| Segmentation threshold | 0.90 |
| Mask binarisation threshold | 0.50 |
| Minimum banana area | 1500 px² |
| Low-confidence warning | < 60% |
| TTA variants | 5 (center crop, h-flip, scale-280, ±10° rotation) |

### Class mapping (fixed, alphabetical)

| Index | Class    | Colour  |
|-------|----------|---------|
| 0     | overripe | Orange  |
| 1     | ripe     | Yellow  |
| 2     | rotten   | Red     |
| 3     | unripe   | Green   |

---

## Frontend Features

- 🎨 Dark glassmorphism UI with animated gradient background
- 🖱 Drag-and-drop image upload with live preview
- 🔄 Animated loading overlay with cycling status messages
- 📊 Animated summary cards with count-up effect
- 🍌 Per-banana detail cards with probability bar charts
- ⚠️ Low-confidence warnings and exposure-correction tags
- ⬇️ Download annotated result image
- 📱 Fully responsive (mobile, tablet, desktop)
- 🔔 Toast notifications for errors

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `503 Models not loaded` | Place `.pth` files in `app/models/` and restart |
| `415 Unsupported file type` | Upload JPG, PNG, or WEBP only |
| CUDA out of memory | Reduce image size before uploading |
| Slow inference | Expected on CPU — GPU strongly recommended |
