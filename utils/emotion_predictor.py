import numpy as np
import cv2
from tensorflow.keras.models import load_model

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

model = load_model("models/emotion_model.hdf5")

def predict_emotion(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img / 255.0
    face_img = np.reshape(face_img, (1, 48, 48, 1))

    preds = model.predict(face_img, verbose=0)
    emotion = emotion_labels[np.argmax(preds)]
    confidence = np.max(preds)

    return emotion, confidence
