# Facial Emotion Recognition System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Build](https://img.shields.io/badge/Build-Stable-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

This project implements a Convolutional Neural Network (CNN) for Facial Emotion Recognition using the FER-2013 dataset.  
The system provides an end-to-end pipeline including model training, evaluation, and real-time inference using a mobile IP Webcam.

The model classifies facial expressions into the following categories:

- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

---

## Dataset

Dataset: FER-2013  
Download: https://www.kaggle.com/datasets/msambare/fer2013  

After downloading and extracting, structure the dataset as follows:

data/
|__ datasets
|
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/

All images must be 48x48 grayscale.

---

## Environment Setup (Windows)

Create virtual environment:

    python -m venv venv

Activate virtual environment:

    venv\Scripts\activate

Install required dependencies:

    pip install tensorflow opencv-python numpy matplotlib

---

## Model Training

To train the CNN model:

    python train.py

Training process includes:

- Loading images using ImageDataGenerator  
- Normalizing pixel values (0–1)  
- Training CNN model  
- Monitoring validation accuracy  
- Saving trained model  

After training, the model is stored as:

    model/emotion_model.h5

---

## Real-Time Emotion Detection (IP Webcam Integration)

Network Requirement:  
Laptop and mobile device must be connected to the same WiFi network.

### Setup Steps

1. Install IP Webcam application on Android device.  
2. Open the application.  
3. Scroll down and select "Start Server".  
4. The app will display an IP address such as:

    http://192.168.1.5:8080

### Configure main.py

Open main.py and update the video capture URL:

    url = "http://192.168.1.5:8080/video"
    cap = cv2.VideoCapture(url)

Replace the IP address with the one displayed in your mobile app.

### Run Live Detection

    python main.py

The system will:

- Capture live video stream  
- Detect faces  
- Predict emotion  
- Display emotion label in real time  

Press 'q' to exit the application.

---

## Model Architecture

The CNN architecture includes:

- Convolutional layers  
- ReLU activations  
- MaxPooling layers  
- Dropout regularization  
- Fully connected dense layers  
- Softmax output layer  

Configuration:

Loss Function: Categorical Crossentropy  
Optimizer: Adam  
Evaluation Metric: Accuracy  

---
## Project Objective

- Implement deep learning-based facial emotion recognition  
- Build a structured training and inference pipeline  
- Deploy real-time prediction system  
- Demonstrate applied computer vision integration  

---

## Author

Vengatesh A
