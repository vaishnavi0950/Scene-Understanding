# 🚗 Scene Understanding for Autonomous Driving
> A multi-task computer vision pipeline for autonomous-driving scene understanding using **semantic segmentation, instance segmentation, object detection, and panoptic fusion**.

---

## 📌 Overview

Autonomous vehicles need to understand their surroundings before making safe driving decisions. A single computer-vision task is often insufficient: identifying *what* is present, *where* it is, and *which pixels belong to which object* are complementary problems.

This project implements a unified **scene-understanding pipeline** that analyzes a road image through multiple deep-learning perception tasks:

* 🟦 **Semantic Segmentation** — classifies pixels into semantic categories.
* 🟩 **Instance Segmentation** — identifies individual object instances and their masks.
* 🟥 **Object Detection** — detects objects using bounding boxes and confidence scores.
* 🟨 **Panoptic Fusion** — combines semantic and instance predictions into a unified scene representation.

The pipeline automatically selects **GPU when CUDA is available** and otherwise falls back to CPU. It can process images placed in the `samples/` directory and produces a visual comparison of the different perception outputs.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Understand the visual environment surrounding an autonomous vehicle.
2. Perform pixel-level semantic understanding of road scenes.
3. Detect and distinguish individual objects.
4. Combine complementary perception outputs into a unified panoptic representation.
5. Build a modular architecture where different perception models can be independently replaced or improved.
6. Provide an end-to-end inference pipeline that can be easily extended to larger autonomous-driving datasets and models.

---

## 🧠 System Architecture

```text
                    Input Road Image
                           │
                           ▼
                 ┌───────────────────┐
                 │   Preprocessing   │
                 └─────────┬─────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
 ┌────────────────┐ ┌───────────────┐ ┌───────────────┐
 │    Semantic    │ │   Instance    │ │    Object     │
 │  Segmentation  │ │  Segmentation │ │   Detection   │
 │                │ │               │ │               │
 │   DeepLabV3    │ │   Mask R-CNN  │ │    YOLOv8     │
 │   ResNet-50    │ │   ResNet-50   │ │      Nano     │
 └───────┬────────┘ └───────┬───────┘ └───────┬───────┘
         │                  │                 │
         │                  ▼                 │
         │          Instance Masks            │
         │          + Classes                 │
         │          + Confidence              │
         │                  │                 │
         └──────────┬───────┘                 │
                    ▼                         │
             ┌───────────────┐                │
             │   Panoptic    │                │
             │    Fusion     │                │
             └───────┬───────┘                │
                     │                        │
                     ▼                        │
              Unified Scene Map               │
                     │                        │
                     └──────────┬─────────────┘
                                ▼
                       ┌─────────────────┐
                       │  Visualization  │
                       └─────────────────┘
                                │
                                ▼
                         result_*.png
```

---

## 🔍 Tasks Implemented

### 1. Semantic Segmentation

Semantic segmentation assigns a class label to every pixel in an image.

This project uses **DeepLabV3 with a ResNet-50 backbone** and pretrained weights. The model produces a dense class map by selecting the highest-scoring class for each pixel.

```text
Input Image
     │
     ▼
DeepLabV3 + ResNet50
     │
     ▼
Pixel-wise Class Predictions
     │
     ▼
Semantic Map
```

Semantic segmentation is useful for understanding scene-level regions such as roads, sky, buildings, vegetation, and other semantic categories.

---

### 2. Instance Segmentation

While semantic segmentation tells us which class each pixel belongs to, instance segmentation additionally distinguishes **different objects belonging to the same class**.

The project uses **Mask R-CNN with a ResNet-50-FPN backbone**. The model produces:

* Bounding boxes
* Class labels
* Confidence scores
* Instance masks

Predictions below the configured confidence threshold are discarded.

```text
Input Image
     │
     ▼
Mask R-CNN
     │
     ├── Bounding Boxes
     ├── Class Labels
     ├── Confidence Scores
     └── Instance Masks
```

---

### 3. Object Detection

Object detection identifies objects using bounding boxes.

The project uses **YOLOv8 Nano (`yolov8n.pt`)**, chosen as a lightweight model suitable for relatively fast inference. The confidence threshold is configurable and is set to `0.5` in the main pipeline.

```text
Input Image
     │
     ▼
YOLOv8n
     │
     ▼
Bounding Boxes + Classes + Confidence
```

YOLOv8 can be replaced with larger variants when accuracy is more important than inference speed.

---

### 4. Panoptic Fusion

Panoptic segmentation combines the ideas of semantic and instance segmentation.

The fusion module starts with the semantic map and overlays confident instance masks to create a unified representation. Each instance receives a unique panoptic ID based on:

```text
panoptic_id = category_id × 1000 + instance_id
```

The resulting representation contains both:

* **Stuff** — background/region-like semantic categories
* **Things** — individual object instances

The fusion module also stores metadata describing each generated segment.

---

## 📊 Why Multiple Tasks?

Each perception task provides different information:

| Task                  | Main Question                                       | Output                   |
| --------------------- | --------------------------------------------------- | ------------------------ |
| Semantic Segmentation | What does each pixel represent?                     | Pixel-wise class map     |
| Instance Segmentation | Which pixels belong to each individual object?      | Masks + classes          |
| Object Detection      | Where are the objects?                              | Bounding boxes + classes |
| Panoptic Fusion       | How can the complete scene be represented together? | Unified panoptic map     |

Combining these tasks provides a richer representation of the driving environment than relying on a single model.

---

## 🗂️ Project Structure

```text
Scene-Understanding/
│
├── data/
│   └── dataset.py
│
├── models/
│   ├── detection.py
│   ├── evaluate.py
│   ├── instance.py
│   ├── metrics.py
│   ├── panoptic.py
│   └── segmentation.py
│
├── samples/
│   └── sample_street.jpg
│
├── utils/
│   └── visualization.py
│
├── main.py
├── dataset.py
├── requirements.txt
├── yolov8n.pt
├── result_0.png
└── README.md
```

### Important files

| File                     | Description                         |
| ------------------------ | ----------------------------------- |
| `main.py`                | Main end-to-end inference pipeline  |
| `data/dataset.py`        | Dataset and inference data loaders  |
| `models/segmentation.py` | DeepLabV3 semantic segmentation     |
| `models/instance.py`     | Mask R-CNN instance segmentation    |
| `models/detection.py`    | YOLOv8 object detection             |
| `models/panoptic.py`     | Semantic + instance panoptic fusion |
| `models/metrics.py`      | Evaluation/metric utilities         |
| `models/evaluate.py`     | Evaluation functionality            |
| `utils/visualization.py` | Visualization of model predictions  |
| `requirements.txt`       | Python dependencies                 |
| `samples/`               | Input street-scene images           |
| `result_*.png`           | Generated pipeline visualizations   |

---

## 🛠️ Technologies Used

### Deep Learning

* PyTorch
* Torchvision
* DeepLabV3
* Mask R-CNN
* YOLOv8

### Computer Vision

* OpenCV
* Pillow

### Data Processing & Visualization

* NumPy
* Matplotlib
* tqdm

The repository's current `requirements.txt` includes PyTorch, Torchvision, Ultralytics, OpenCV, Matplotlib, NumPy, Pillow, and tqdm.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/vaishnavi0950/Scene-Understanding.git
cd Scene-Understanding
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

Run the main pipeline:

```bash
python main.py
```

The program will:

1. Check whether CUDA is available.
2. Initialize the perception models.
3. Download a sample street image if one is not already present.
4. Load the image from `samples/`.
5. Run semantic segmentation.
6. Run instance segmentation.
7. Run YOLOv8 object detection.
8. Fuse semantic and instance predictions.
9. Generate visualizations.
10. Save the final result as `result_0.png`.

The current implementation explicitly performs these four perception stages sequentially in `main.py`.

---

## 🖼️ Using Your Own Images

You can place your own road-scene images inside:

```text
samples/
```

For example:

```text
samples/
├── highway.jpg
├── city_road.jpg
├── intersection.jpg
└── traffic.jpg
```

Then run:

```bash
python main.py
```

The inference loader is designed to read street images placed in the `samples` directory.

---

## 💻 CPU vs GPU

The pipeline automatically determines the available device:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

Therefore:

* **NVIDIA GPU + CUDA available** → GPU inference
* **No CUDA GPU available** → CPU inference

A CUDA-capable GPU is recommended for significantly faster inference, particularly because the pipeline executes multiple deep-learning models for every image.

---

## 📚 Dataset

The project includes a dataset wrapper designed around the **Cityscapes dataset**.

The wrapper expects the standard Cityscapes components such as:

```text
leftImg8bit/
gtFine/
```

and supports the native `torchvision.datasets.Cityscapes` interface.

For demonstration/inference purposes, the project does not require the full Cityscapes dataset. Instead, images can be placed directly in the `samples/` directory.

### Cityscapes

Cityscapes is particularly relevant to this project because it contains urban street scenes captured from a vehicle-mounted perspective and provides detailed annotations for scene understanding.

> **Note:** The current GitHub implementation is primarily configured for inference/demo usage. The dataset wrapper provides the foundation for working with Cityscapes, while the main pipeline uses sample images for inference.

---

## 🔄 End-to-End Pipeline

For each input image, the pipeline follows:

### Step 1 — Input

```text
Road Scene Image
```

### Step 2 — Semantic Understanding

```text
Image → DeepLabV3 → Semantic Map
```

### Step 3 — Instance Understanding

```text
Image → Mask R-CNN → Object Masks
```

### Step 4 — Object Detection

```text
Image → YOLOv8 → Bounding Boxes
```

### Step 5 — Panoptic Fusion

```text
Semantic Map
      +
Instance Masks
      ↓
Panoptic Scene Representation
```

### Step 6 — Visualization

The final visualization displays the original image alongside the outputs of the different perception branches.

---

## 📈 Evaluation

The repository contains metric/evaluation utilities under:

```text
models/
├── metrics.py
└── evaluate.py
```

These components can be extended to evaluate segmentation and detection performance using task-specific metrics.

Typical metrics for this type of system include:

* **IoU / Intersection over Union**
* **Mean IoU (mIoU)**
* **Pixel Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **Detection confidence**
* **Panoptic Quality (PQ)**

For a fully trained autonomous-driving perception system, these metrics should be evaluated against ground-truth annotations from datasets such as Cityscapes.

---

## 🎨 Output

The pipeline generates a result image such as:

```text
result_0.png
```

The visualization contains the different perception results, allowing the outputs of semantic segmentation, instance segmentation, object detection, and panoptic fusion to be compared for the same scene.

---

## ⭐ Key Features

* ✅ Multi-task autonomous-driving perception
* ✅ Semantic segmentation with DeepLabV3
* ✅ Instance segmentation with Mask R-CNN
* ✅ Object detection with YOLOv8
* ✅ Panoptic scene fusion
* ✅ Automatic CPU/GPU selection
* ✅ Modular model architecture
* ✅ Custom street-image inference
* ✅ Visualization of intermediate and final outputs
* ✅ Cityscapes-compatible dataset wrapper
* ✅ Easily replaceable perception models

---

## 🔧 Design Decisions

### Why DeepLabV3?

DeepLabV3 provides a strong semantic segmentation architecture capable of producing dense pixel-level predictions while leveraging a pretrained ResNet backbone.

### Why Mask R-CNN?

Mask R-CNN provides both object localization and pixel-level instance masks, making it suitable for identifying individual objects in complex road scenes.

### Why YOLOv8?

YOLOv8 provides fast object detection and is useful when bounding-box-based object localization is required alongside more detailed segmentation outputs.

### Why Panoptic Fusion?

Semantic and instance segmentation solve complementary problems. Fusion allows the system to represent both region-level scene context and individual objects in one unified map.

---

## ⚠️ Current Limitations

The current implementation is intended as a **modular scene-understanding prototype/inference pipeline**, rather than a production-ready autonomous-driving perception system.

Some limitations include:

* The models use pretrained generic weights rather than being specifically fine-tuned on a common autonomous-driving dataset.
* The semantic segmentation model is based on generic pretrained DeepLabV3 weights.
* The instance segmentation model is based on generic pretrained Mask R-CNN weights.
* YOLOv8 uses a general-purpose pretrained model.
* Panoptic fusion is a lightweight fusion strategy and is not equivalent to a benchmarked end-to-end panoptic segmentation model.
* Full Cityscapes training/evaluation is not part of the current `main.py` inference flow.
* Real autonomous-driving deployment would require additional considerations such as temporal consistency, sensor fusion, latency optimization, uncertainty estimation, and safety validation.

---

## 🚧 Future Improvements

Potential extensions include:

* [ ] Fine-tune DeepLabV3 on Cityscapes.
* [ ] Fine-tune Mask R-CNN on autonomous-driving datasets.
* [ ] Train YOLOv8 specifically on road-scene objects.
* [ ] Improve panoptic fusion using proper thing/stuff class mappings.
* [ ] Add quantitative Cityscapes evaluation.
* [ ] Add real-time webcam/video inference.
* [ ] Add temporal tracking between consecutive frames.
* [ ] Add depth estimation.
* [ ] Add lane detection.
* [ ] Add traffic-sign and traffic-light recognition.
* [ ] Add multi-camera or LiDAR fusion.
* [ ] Optimize models for real-time edge deployment.
* [ ] Export models to ONNX/TensorRT for optimized inference.

---

## 🧪 Example Use Cases

This project can serve as a foundation for:

* Autonomous-driving perception
* Advanced Driver Assistance Systems (ADAS)
* Road-scene analysis
* Intelligent transportation systems
* Computer-vision research
* Multi-task learning experiments
* Semantic and instance segmentation research
* Panoptic scene understanding

---

## 📖 References

* Cityscapes Dataset — *The Cityscapes Dataset for Semantic Urban Scene Understanding*
* DeepLabV3 — *Rethinking Atrous Convolution for Semantic Image Segmentation*
* Mask R-CNN — *Mask R-CNN*
* YOLO — *You Only Look Once: Unified, Real-Time Object Detection*
* PyTorch / Torchvision
* Ultralytics YOLO

---

## 👩‍💻 Author

**Vaishnavi Reddy**

B.Tech — Computer Science and Engineering
Indian Institute of Technology Patna

---

## 📄 License

This project is intended for educational and research purposes.

Please check the licenses of the underlying pretrained models, datasets, and third-party libraries before using this project for commercial applications.

---

## ⭐ Acknowledgements

This project builds upon the open-source deep-learning ecosystem provided by:

* PyTorch
* Torchvision
* Ultralytics
* OpenCV
* Cityscapes

---

## 🔗 Repository

**GitHub:**
https://github.com/vaishnavi0950/Scene-Understanding
