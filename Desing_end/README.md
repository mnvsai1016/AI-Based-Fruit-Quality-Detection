# 🍌 BananaVision — Technical Documentation & Pipeline Guide

This folder contains the core application backend, machine learning pipeline, training notebooks, and the web interface for the **AI-Based Fruit Quality Detection** system (currently specialized for bananas).

---

## 🛠️ Architecture & Pipeline Overview

The pipeline executes a two-stage deep learning approach to accurately segment and classify bananas from ordinary photos:

```
[Input Image] ───► [Mask R-CNN ResNet-50 FPN] ───► [Banana Crops + Masks]
                                                            │
                                                            ▼
                                           [MobileNetV2 Classifier + TTA]
                                                            │
                                                            ▼
                                              [Ripeness Class & Probabilities]
                                                            │
                                                            ▼
                                              [Shelf-Life & Storage Insights]
```

### Key Parameters:
- **Segmentation Confidence Threshold**: `0.90` (Mask R-CNN)
- **Mask Binarization Threshold**: `0.50`
- **Minimum Banana Area Filter**: `1500 px²`
- **Low-Confidence Warning**: Confidence `< 60%`
- **Test-Time Augmentation (TTA)**: 5 deterministic variants (Center Crop, Horizontal Flip, 280px Scale, +10° Rotation, -10° Rotation)

---

## 📂 Directory Layout

```
Desing_end/
├── app/
│   ├── main.py                              # FastAPI web server and REST endpoints
│   ├── pipeline.py                          # Two-stage segmentation & classification pipeline
│   ├── models/                              # Trained .pth model weight files
│   │   ├── banana_maskrcnn_finetuned.pth    # Fine-tuned Mask R-CNN model
│   │   └── banana_mobilenetv2_best.pth      # Best MobileNetV2 classifier
│   └── static/
│       ├── index.html                       # Responsive HTML5/CSS3 frontend
│       └── samples/                         # Built-in sample test images
├── BANANA_FINAL_INTEGRATED.ipynb            # Integration & end-to-end testing notebook
├── Banana_Segmentation_training_code.ipynb  # Mask R-CNN training on custom annotated dataset
├── banana_ripeness_training_code.ipynb      # MobileNetV2 transfer learning & evaluation
├── banana_pipeline_integration.ipynb        # Pipeline prototyping notebook
└── requirements.txt                         # Application dependencies
```

---

## 🚀 Running the Local Server

```bash
# 1. Navigate to the app directory
cd app

# 2. Start the Uvicorn server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then visit **http://localhost:8000** in your browser.

---

## 🔌 API Endpoints

- **`GET /health`**: Healthcheck endpoint returning server status, hardware acceleration (`cuda`/`cpu`), and model status.
- **`GET /samples`**: Returns a list of built-in preset demo images with metadata.
- **`POST /analyze`**: Accepts a multipart image file (`JPG`/`PNG`/`WEBP`) and returns bounding boxes, segmentation masks, ripeness classification, confidence scores, and storage recommendations in JSON format.
