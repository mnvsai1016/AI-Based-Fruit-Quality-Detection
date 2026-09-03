# 🍌 BananaVision — AI-Based Fruit Quality & Ripeness Detection

An end-to-end computer vision and deep learning project designed to automate fruit quality inspection, instance detection, and ripeness grading.

---

## 📌 Project Overview & Scope

Quality inspection of fresh produce is critical across supply chains, retail stores, and smart kitchens. This repository represents our work on an **AI-Based Fruit Quality Detection System**. 

> **Current Scope & Focus**:  
> In its current version, this project is fully developed and fine-tuned specifically for **Bananas**. It detects individual bananas (both single fruits and clustered bunches), segments each banana from the background, classifies its ripeness stage across four categories (**Unripe**, **Ripe**, **Overripe**, and **Rotten**), and provides estimated shelf-life with practical storage and culinary advice.  
> 
> The system is built with a modular architecture so the pipeline can be extended to additional fruit varieties (such as apples, mangoes, and oranges) in future stages.

---

## 🔬 How It Works (Two-Stage Pipeline)

```
                       [ Input Image / Live Photo ]
                                    │
                                    ▼
                 ┌──────────────────────────────────────┐
                 │    Stage 1: Instance Segmentation    │
                 │       (Mask R-CNN ResNet-50 FPN)     │
                 └──────────────────┬───────────────────┘
                                    │
                                    ▼  Extracts isolated banana masks & crops
                 ┌──────────────────────────────────────┐
                 │    Stage 2: Ripeness Classification  │
                 │       (MobileNetV2 + 5-Fold TTA)     │
                 └──────────────────┬───────────────────┘
                                    │
                                    ▼  Predicts ripeness probabilities
                 ┌──────────────────────────────────────┐
                 │    Stage 3: Fruit Intelligence Engine│
                 │ (Shelf Life, Quality Score, Advice)  │
                 └──────────────────┬───────────────────┘
                                    │
                                    ▼
                 [ Interactive Visual Quality Report ]
```

### 1. Instance Segmentation (Mask R-CNN)
Detects individual bananas within an image, distinguishes overlapping bananas in a bunch, and produces precise pixel-level masks and bounding boxes.

### 2. Ripeness Classification (MobileNetV2 + Test-Time Augmentation)
Each segmented banana is cropped, background-masked, and passed through a fine-tuned MobileNetV2 network with 5 deterministic Test-Time Augmentation (TTA) transforms to ensure robust predictions under varying lighting and camera angles.

### 3. Ripeness Categories & Quality Insights

| Ripeness Stage | Visual Characteristics | Freshness Rating | Shelf Life | Culinary / Practical Use |
| :--- | :--- | :---: | :---: | :--- |
| 🟢 **Unripe** | Green peel, firm texture | 95% | 5–7 Days | High resistant starch; ideal for cooking and longer storage |
| 🟡 **Ripe** | Golden yellow, sweet pulp | 90% | 3–5 Days | Peak flavor; ready for direct snacking, fruit bowls, and salads |
| 🟠 **Overripe** | Brown sugar spots, soft texture | 65% | 1–2 Days | Maximum sweetness; best for banana bread, pancakes, smoothies |
| 🔴 **Rotten** | Darkened skin, mushy/fermented | 10% | 0 Days | Spoiled; unsuitable for consumption (use for organic compost) |

---

## 📁 Repository Structure

```
├── Desing_end/
│   ├── app/
│   │   ├── main.py                              # FastAPI backend application
│   │   ├── pipeline.py                          # Dual-inference pipeline engine
│   │   ├── models/                              # Pretrained weights (.pth files)
│   │   └── static/
│   │       ├── index.html                       # Responsive web UI
│   │       └── samples/                         # Built-in demo sample images
│   ├── BANANA_FINAL_INTEGRATED.ipynb            # End-to-end integration notebook
│   ├── Banana_Segmentation_training_code.ipynb  # Mask R-CNN segmentation training
│   ├── banana_ripeness_training_code.ipynb      # MobileNetV2 classification training
│   ├── banana_pipeline_integration.ipynb        # Pipeline evaluation notebook
│   ├── requirements.txt                         # Full Python dependencies
│   └── README.md                                # Detailed backend technical notes
├── requirements.txt                             # Root project dependencies
└── README.md                                    # Main project documentation
```

---

## 💻 Getting Started (Local Setup)

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AI-Based-Fruit-Quality-Detection.git
cd AI-Based-Fruit-Quality-Detection
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note for GPU Acceleration (CUDA)**:  
> If you have an NVIDIA GPU and want real-time deep learning inference, install PyTorch with CUDA:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Run the Web Application
```bash
cd Desing_end/app
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🌐 Using the Application

- **On Desktop**: Open [http://localhost:8000](http://localhost:8000) in your web browser.
- **On Mobile Phone (Same Wi-Fi network)**: Open `http://<YOUR_PC_IP>:8000` (e.g. `http://192.168.1.X:8000`) on Chrome/Safari.

### Web Interface Highlights:
- ⚡ **1-Click Test Samples**: Test instant detection with built-in presets (Unripe, Ripe, Overripe, and Bunch) without needing to take photos.
- 📸 **Live Camera Snap**: Live camera viewfinder with front $\leftrightarrow$ rear lens flipping for instant inspection on phone or webcam.
- 📊 **Confidence & Probabilities**: Animated probability breakdown charts for each detected banana.
- 💡 **Storage & Freshness Advice**: Context-aware shelf-life estimates and recipes.
- 📲 **Native Sharing**: Share detection summaries directly to messaging apps.

---

## 🔮 Future Roadmap

- 🍎 **Multi-Fruit Expansion**: Training and extending the pipeline to evaluate quality for apples, mangoes, tomatoes, and citrus fruits.
- 📦 **Defect & Bruise Detection**: Segmenting external surface blemishes, mechanical bruises, and fungal spots.
- ☁️ **Cloud Deployment**: Containerizing the model for scalable cloud hosting and mobile edge integration.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
