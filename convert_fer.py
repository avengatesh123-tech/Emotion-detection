import pandas as pd
import numpy as np
import cv2
import os

emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

data = pd.read_csv("data/fer2013.csv")

for index, row in data.iterrows():
    emotion = emotion_map[row["emotion"]]
    pixels = row["pixels"]
    usage = row["Usage"]

    img = np.array(pixels.split(), dtype="uint8")
    img = img.reshape(48, 48)

    if usage == "Training":
        folder = "data/dataset/train/" + emotion
    else:
        folder = "data/dataset/test/" + emotion

    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(f"{folder}/{index}.jpg", img)

print("FER dataset converted.")
