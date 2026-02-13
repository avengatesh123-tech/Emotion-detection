import cv2
import time
import csv
from utils.face_detector import detect_faces
from utils.emotion_predictor import predict_emotion

camera_url = "http://10.46.89.18:8080/video"  

cap = cv2.VideoCapture(camera_url)

log_file = open("data/logs.csv", "a", newline="")
csv_writer = csv.writer(log_file)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not received")
        break

    faces, gray = detect_faces(frame)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        emotion, confidence = predict_emotion(face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{emotion} ({confidence:.2f})"

        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([timestamp, emotion, confidence])

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
