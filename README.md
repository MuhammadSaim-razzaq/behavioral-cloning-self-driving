# 🚗 Self-Driving Car Simulator

An end-to-end deep learning project that trains a convolutional neural network (CNN) to autonomously drive a car in the Udacity Self-Driving Car Simulator using **behavioral cloning**. Inspired by NVIDIA's landmark paper *End to End Learning for Self-Driving Cars* (Bojarski et al., 2016).

---

## 📋 Table of Contents

- [Overview](#overview)
- [How This Project Works](#how-this-project-works)
- [Project Structure](#project-structure)
- [Training on Google Colab](#training-on-google-colab)
- [Local Environment Setup](#local-environment-setup)
- [Running the Simulator](#running-the-simulator)
- [Model Architecture](#model-architecture)
- [Data Processing & Augmentation](#data-processing--augmentation)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## 🎯 Overview

This project implements autonomous driving using the end-to-end learning paradigm — raw camera pixels are mapped directly to steering angle predictions, without any separate lane detection, path planning, or handcrafted rules.

**The workflow is split across two environments:**

1. **Google Colab** — Model training using a GPU runtime on collected simulator data
2. **Local Machine** — Real-time inference with `drive.py`, which connects to the Udacity simulator and steers the car autonomously

---

## 🔄 How This Project Works

```
[Udacity Simulator - Training Mode]
         |
         | Manual driving → saves driving_log.csv + images
         ▼
[Google Colab - Training]
         |
         | Train CNN on collected data → exports model.h5
         ▼
[Local Machine - Inference]
         |
         | drive.py loads model.h5 → Socket.IO server on port 4567
         ▼
[Udacity Simulator - Autonomous Mode]
         |
         | Sends camera frames → receives steering + throttle commands
```

---

## 📁 Project Structure

```
selfDrivingCar/
├── 📓 train.ipynb          # Google Colab training notebook
├── 🚗 drive.py             # Local inference + simulator interface
├── 🤖 model.h5             # Trained model weights (exported from Colab)
└── 📁 data/                # Training data (used in Colab)
    ├── driving_log.csv     # Steering, throttle, brake, speed logs
    └── IMG/                # Center, left, right camera images
```

---

## ☁️ Training on Google Colab

All model training happens in Google Colab to leverage free GPU acceleration. No local GPU is required.

### Step 1 — Collect Data

1. Download and launch the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)
2. Select **Training Mode** and drive manually for several laps
3. The simulator saves `driving_log.csv` and an `IMG/` folder with camera images
4. Compress your data folder: `data.7z`

### Step 2 — Upload to Google Drive

Upload your `data.7z` archive to your Google Drive.

### Step 3 — Run the Notebook

Open `train.ipynb` in Google Colab. The notebook will:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Extract the dataset
!apt-get install p7zip-full
!7z x "/content/drive/MyDrive/data.7z" -o/content/myData
```

Then it runs through the full pipeline:
- Load and balance steering angle distribution
- Split into training/validation sets (80/20)
- Apply real-time data augmentation during training
- Train the NVIDIA-inspired CNN for 10 epochs
- Save `model.h5` and plot the loss curve

### Step 4 — Download the Model

```python
from google.colab import files
files.download("model.h5")
```

Place the downloaded `model.h5` in the same folder as `drive.py` on your local machine.

---

## 💻 Local Environment Setup

> ⚠️ **Important:** The simulator requires very specific versions of `python-socketio` and `python-engineio`. Using the wrong versions will cause connection failures. A dedicated conda environment is strongly recommended.

### Step 1 — Create a Clean Conda Environment

```bash
# Deactivate any existing environment
conda deactivate

# Create a new Python 3.9 environment
conda create -n drive_env python=3.9 -y

# Activate it
conda activate drive_env
```

### Step 2 — Install Dependencies

```bash
# Core ML and utility packages
pip install tensorflow scikit-learn numpy pandas matplotlib pillow opencv-python imgaug flask eventlet

# Strict networking versions required for simulator compatibility
pip install python-engineio==3.13.2 python-socketio==4.6.1
```

> These exact versions of `python-socketio` and `python-engineio` are required. Newer versions use a different API incompatible with the simulator's communication protocol.

---

## 🎮 Running the Simulator

### Step 1 — Start the Drive Server

With your conda environment active and `model.h5` in the same directory as `drive.py`:

```bash
conda activate drive_env
python drive.py
```

You should see:
```
Setting UP
(4567) wsgi starting up on http://0.0.0.0:4567
```

### Step 2 — Launch the Simulator

1. Open the Udacity simulator
2. Select a track
3. Choose **Autonomous Mode**
4. The simulator connects to `localhost:4567`

Once connected, the terminal will display live telemetry:
```
Connected
Steering: -0.0231, Throttle: 0.8143, Speed: 1.74
Steering:  0.0104, Throttle: 0.7921, Speed: 2.31
...
```

The car will begin driving autonomously.

---

## 🏗️ Model Architecture

The model is based on the NVIDIA end-to-end CNN architecture. Raw YUV images (66×200×3) are fed directly into the network, which outputs a single steering angle.

```
Input: 66 × 200 × 3 (YUV image)
│
├── Conv2D(24, 5×5, stride=2, ELU)
├── Conv2D(36, 5×5, stride=2, ELU)
├── Conv2D(48, 5×5, stride=2, ELU)
├── Conv2D(64, 3×3, ELU)
├── Conv2D(64, 3×3, ELU)
│
├── Flatten
├── Dense(100, ELU)
├── Dense(50, ELU)
├── Dense(10, ELU)
└── Dense(1, linear)  →  steering angle
```

**Why ELU?** Exponential Linear Units avoid the dying neuron problem of ReLU and produce smoother gradients, which benefits regression tasks like steering prediction.

**Key design principle:** The network has no explicit lane detection or feature engineering. It learns to extract all relevant road features — lane boundaries, curves, road edges — purely from the steering supervision signal.

---

## 🔄 Data Processing & Augmentation

### Preprocessing Pipeline

Applied to every image before it enters the model (both training and inference):

```python
def preProcess(img):
    img = img[60:135, :, :]           # Crop sky and car hood
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # RGB → YUV
    img = cv2.GaussianBlur(img, (3,3), 0)       # Smooth noise
    img = cv2.resize(img, (200, 66))             # Resize to model input
    img = img / 255                              # Normalize to [0, 1]
    return img
```

### Data Augmentation (Training Only)

Applied randomly during batch generation to improve generalization:

| Technique | Probability | Effect |
|---|---|---|
| Random pan | 50% | Simulates lateral road position variation |
| Zoom (1.0–1.2×) | 50% | Simulates distance variation |
| Brightness adjustment | 50% | Simulates different lighting conditions |
| Horizontal flip | 50% | Doubles data; steering angle is negated |

### Data Balancing

Raw driving data is heavily biased toward `steering = 0` (straight driving). The balancing step caps each steering bin at 500 samples to prevent the model from learning to always go straight.

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Mean Squared Error (MSE) |
| Batch size | 64 |
| Epochs | 10 |
| Steps per epoch | 300 |
| Validation steps | 200 |
| Train/val split | 80% / 20% |

Augmentation is applied **only** to training batches. Validation batches use raw images to give an unbiased performance estimate.

---

## 📊 Results

- Validation loss decreases steadily over 10 epochs with no sign of overfitting
- The car completes full laps autonomously on the training track
- Real-time inference runs smoothly; the server processes frames and sends control commands with minimal latency
- Speed is capped at `maxSpeed = 10` with dynamic throttle: `throttle = 1.0 - speed / maxSpeed`

---

## 📦 Dependencies

### Colab (Training)

```
tensorflow
opencv-python
pandas
numpy
matplotlib
scikit-learn
albumentations
pillow
```

### Local (Inference)

```
tensorflow
opencv-python
numpy
pillow
flask
eventlet
python-socketio==4.6.1      ← exact version required
python-engineio==3.13.2     ← exact version required
```

---

## 🔧 Troubleshooting

**Simulator not connecting**
- Confirm `drive.py` is running and listening on port 4567 before launching autonomous mode
- Check that `python-socketio==4.6.1` and `python-engineio==3.13.2` are installed in the active environment

**Model weights not loading**
- Confirm `model.h5` is in the same directory as `drive.py`
- `drive.py` rebuilds the model architecture locally and calls `model.load_weights('model.h5')` — this avoids Keras version compatibility issues between Colab and your local machine

**Car drives off the road**
- Collect more diverse training data (vary speed, hug both sides of the lane)
- Ensure the data balancing threshold is appropriate for your dataset size
- Try more training epochs or a lower learning rate

**`ImportError` or version conflicts**
- Start fresh: `conda create -n drive_env python=3.9 -y` and reinstall all packages
- Do not upgrade `python-socketio` or `python-engineio` — newer versions break simulator compatibility

**Slow inference / lag**
- Reduce `maxSpeed` in `drive.py` to give the model more reaction time
- Make sure you are using a CPU-optimised TensorFlow build locally

---

## 📚 References

- Bojarski, M. et al. (2016). [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316). NVIDIA.
- [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)
- [Albumentations — fast image augmentation library](https://albumentations.ai/)
