from flask import Flask, render_template, Response
import cv2
import torch
import numpy as np
import pygame
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model (Replace with your model file)
model = YOLO("p91.pt")  # Your trained YOLO model for person detection

# Initialize Pygame for sound
pygame.mixer.init()
sound = pygame.mixer.Sound("ktm.mp3")  # Load alert sound

# Open webcam
cap = cv2.VideoCapture(0)
person_detected = False  # Track if a person is detected

def detect_person():
    global person_detected
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # Run YOLO on the frame
        detected = False  # Flag for person detection

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                if class_id == 0:  # Person class
                    detected = True  # Person found
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Play sound if person is detected, stop otherwise
        if detected and not person_detected:
            sound.play(-1)  # Loop sound
            person_detected = True
        elif not detected and person_detected:
            sound.stop()
            person_detected = False

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_person(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
