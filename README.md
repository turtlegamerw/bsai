
# 🧠 Brawl Stars Gadget Detection Bot

This project is a **real-time object detection bot for Brawl Stars**, powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).  
It detects UI elements such as when a gadget is charged and automatically simulates a tap using ADB on your Android device.

## 🎮 What It Does

- Mirrors your game screen using `scrcpy`
- Captures the screen using `mss` in real-time
- Runs YOLOv8 on each frame to detect objects like:
  - `gadget_charged`
  - `gadget_not_charged`
- Sends tap commands using pure Python ADB integration when actions are triggered

---

## 🧩 Components

```
project/
├── gadget.py         # Tap coordinates for gadgets
├── test.py           # Main detector and action loop
├── best.pt           # Trained YOLOv8 model
├── data.yaml         # Dataset configuration file
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
```

---

## ⚙️ How to Train a Model

### 1. Prepare Your Data

Use a tool like [makesense.ai](https://makesense.ai) or [CVAT](https://cvat.org) to annotate screenshots from Brawl Stars.

You must export using:

> ✅ **YOLO with Images**

Structure:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
data.yaml
```

### 2. Create `data.yaml`

```yaml
train: dataset/train/images
val: dataset/val/images
nc: 2
names: ['gadget_charged', 'gadget_not_charged']
```

### 3. Train the Model

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=30 imgsz=640 device=0
```

---

## ❌ What Went Wrong

### ❗ Labeling Took Too Long

The project didn’t progress as planned because the **manual labeling process was too slow and time-consuming**. While the tools worked correctly (like makesense.ai), the effort required to label enough examples for training made it difficult to build a large, high-quality dataset.

**Why It Failed:**
- Labeling each image manually became tedious and repetitive
- Not enough labeled data to train a reliable YOLOv8 model
- Dataset remained too small for good detection performance

### 🕐 Why It Didn’t Work Out

Even after organizing the dataset and setting up training, the combination of:
- Manual labeling overhead
- Limited compute power (GTX 1650 GPU)
- Small dataset

...led to slow progress. This blocked testing and real-time integration into the bot.

**Possible solutions:**
- Use Google Colab for training
- Use fewer epochs or a smaller model (`yolov8n`)
- Collaborate to speed up labeling

---

## 🚀 Future Ideas

- Add detection for enemies, power cubes, and supers
- Implement smarter action logic (e.g., auto-aim)
- Use segmentation or tracking models for more precise control

---

## 🙌 Credits

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- `scrcpy` for screen mirroring
- `mss` for screen capture
- ADB for Android input simulation
- Chatgpt for making this :)
