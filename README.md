# Emotion Detection System

A real-time facial emotion recognition system built using computer vision and deep learning.
The application detects human faces from a webcam feed and predicts emotions in real time using a trained neural network model.

---

## Overview

This project implements an end-to-end emotion detection pipeline:

1. Face detection from live video input
2. Emotion classification using a trained model
3. Real-time visualization of predictions
4. Logging of detected emotions for analysis

The system is designed to be modular, easy to extend, and suitable for experimentation or deployment.

---

## Key Features

* Real-time emotion detection via webcam
* Deep learning–based emotion classification
* Modular architecture for easy customization
* Training pipeline for emotion recognition models
* Dataset conversion utility
* Emotion logging for analysis and debugging

---

## Tech Stack

* **Language:** Python
* **Computer Vision:** OpenCV
* **Deep Learning:** TensorFlow / Keras
* **Data Handling:** NumPy, Pandas

---



## Installation

### 1. Clone the repository

```bash
git clone https://github.com/avengatesh123-tech/Emotion-detection.git
cd Emotion-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the **FER (Facial Expression Recognition)** dataset.

To convert the dataset into the required format:

```bash
python convert_fer.py
```
---

## Usage

### Train the emotion detection model

```bash
python train_model.py
```

This will:

* Train the model on the dataset
* Save the trained model in the `models/` directory

---

### Run real-time emotion detection

```bash
python main.py
```

This will:

* Start the webcam
* Detect faces
* Display predicted emotions in real time
---

## Output

* Real-time emotion predictions displayed on screen
* Emotion logs stored in:

```
data/logs.csv
```

---

## Possible Improvements

* Web interface for emotion analytics
* Cloud deployment (AWS/GCP/Azure)
* REST API for emotion prediction
* Improved model accuracy with advanced architectures
* Mobile or browser-based integration

---

## Author

**Vengatesh A**

---

## License

This project is intended for educational and research purposes...
