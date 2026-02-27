Facial Emotion Recognition System (CNN + Live Camera)
Project Overview

This project implements a Convolutional Neural Network (CNN) for Facial Emotion Recognition using the FER-2013 dataset.

The system can:

Train a CNN model on emotion images

Evaluate model performance

Run real-time emotion detection using mobile IP Webcam

The model classifies facial expressions into:

Angry

Disgust

Fear

Happy

Neutral

Sad

Surprise

Dataset

Dataset used: FER-2013

Download from Kaggle:

https://www.kaggle.com/datasets/msambare/fer2013

After extraction, organize the dataset as:

data/
   datasets/
   train/
      angry/
      disgust/
      fear/
      happy/
      neutral/
      sad/
      surprise/
   test/
      angry/
      disgust/
      fear/
      happy/
      neutral/
      sad/
      surprise/

Each folder must contain 48x48 grayscale images.

Environment Setup (Windows)
Step 1: Create Virtual Environment
python -m venv venv
Step 2: Activate Environment
venv\Scripts\activate
Step 3: Install Dependencies
pip install tensorflow opencv-python numpy matplotlib
Training the Model

Run:

python train.py

What happens internally:

Images loaded using ImageDataGenerator

Rescaled to 0–1

CNN model is trained

Validation accuracy is monitored

Model is saved as:

model/emotion_model.h5

Training may take several minutes depending on system performance.

Running Real-Time Emotion Detection (IP Webcam)

This project supports live detection using your mobile camera.

Step 1: Install IP Webcam App

Install IP Webcam app from Play Store on your Android device.

Step 2: Connect to Same WiFi

Your:

Laptop

Mobile phone

Must be connected to the SAME WiFi network.

If not, it will not work.

Step 3: Start Server in IP Webcam

Open IP Webcam app

Scroll down

Click Start Server

App will show an IP address like:

http://192.168.1.5:8080

This is your mobile camera stream.

Step 4: Copy IP Address

Copy only the base address:

http://192.168.1.5:8080
Step 5: Add IP in main.py

Open main.py and modify the video capture line:

url = "http://192.168.1.5:8080/video"
cap = cv2.VideoCapture(url)

Replace the IP with your mobile’s IP.

Important:
The IP must match exactly what the app shows.

Step 6: Run Live Detection
python main.py

Now:

Your laptop connects to mobile camera

Face is detected

Emotion is predicted in real-time

Output label is displayed on screen

Press q to exit.

Model Details

Architecture includes:

Convolution Layers

ReLU Activation

MaxPooling

Dropout

Dense Layers

Softmax Output

Loss Function: Categorical Crossentropy
Optimizer: Adam
Metric: Accuracy

Project Objective

Learn CNN-based image classification

Understand real-time model deployment

Integrate mobile camera streaming with Python

Build an end-to-end emotion recognition system

Author

Vengatesh A
