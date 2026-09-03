# 🍌 BananaVision — AI-Based Fruit Quality & Ripeness Detection

An AI-powered fruit quality detection system that uses Computer Vision & Deep Learning to automatically detect fruit instances, assess freshness, and classify ripeness stages in real time.

> **Note on Scope**: This project is part of the **AI-Based Fruit Quality Detection** system and is **currently implemented and fine-tuned specifically for Bananas** (supporting *Unripe*, *Ripe*, *Overripe*, and *Rotten/Decayed* classification, individual instance segmentation, shelf-life estimation, and culinary advice). The underlying pipeline is modular and designed to be expanded to other fruits in subsequent phases.

---

## 🌟 Key Features

- 🍌 **Banana-Specific Ripeness Engine**: Detects individual bananas in single or bunch formats and classifies ripeness across 4 key stages (*Unripe, Ripe, Overripe, Rotten*).
- 📱 **Mobile Browser Friendly**: Responsive viewport with thumb-friendly controls, camera access, and fluid layouts for smartphones.
- 📸 **Live In-Browser Camera**: Live viewfinder modal with front/back camera flipping and snapshot capture.
- ⚡ **1-Click Test Demos**: Built-in sample presets for instant testing without uploading files.
- 🥗 **Fruit Shelf-Life & Storage Guidance**: Estimated remaining days, freshness score, storage tips, and recipes (smoothies, baking, snacking).
- 🎨 **Signature Dark Theme**: High-contrast glassmorphism UI with color-coded fruit badges.
- 🛡️ **Dual Inference Engine**: Automatically uses Deep Learning (Mask R-CNN + MobileNetV2) when `.pth` model weights are loaded, and falls back to an intelligent Computer Vision engine when deployed on serverless cloud platforms (Vercel).

---

## 🏗️ Technical Architecture (Banana Pipeline)

```
Image Input → Mask R-CNN Instance Segmentation → Background Masked Crop
                                                        ↓
                                         MobileNetV2 + 5-Transform TTA
                                                        ↓
                               Ripeness: Unripe / Ripe / Overripe / Rotten
                                                        ↓
                             Quality Score + Shelf-Life + Storage Advice
```

| Component | Model / Method | Output |
| :--- | :--- | :--- |
| **Detection & Segmentation** | Mask R-CNN ResNet-50 FPN | Individual banana masks, bounding boxes, detection scores |
| **Ripeness Classification** | MobileNetV2 (1280 → 4) + TTA | Ripeness class, confidence %, probability distribution |
| **Actionable Guidance** | Rule-based Expert Engine | Shelf-life remaining, storage tips, culinary recommendations |

---

## 🚀 Quick Start (Local Setup)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
cd Desing_end/app
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open in Browser
- **Desktop**: [http://localhost:8000](http://localhost:8000)
- **Mobile (Same Wi-Fi)**: `http://<YOUR-PC-IP>:8000` (e.g. `http://172.17.196.158:8000`)

---

## ☁️ Deploy to Vercel

1. **Push this repository to GitHub**.
2. Go to [Vercel Dashboard](https://vercel.com/dashboard) $\rightarrow$ **"Add New Project"**.
3. Import this GitHub repository.
4. Click **"Deploy"** (Vercel automatically detects `vercel.json` and builds the app).

---

## 📄 License
MIT License - Open source & free to use.
