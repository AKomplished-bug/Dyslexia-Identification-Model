import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import time

# Load the trained model
model = load_model("../Data/saved_model/dyslexia_detection_model.h5")
print("Model loaded successfully.")

# scaler :)
scaler = StandardScaler()

# Parameters
time_steps = 100  
num_features = 4  
sequence = []

# MediaPipe 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_eye_tracking_data(frame):
    # Convert the frame to RGB (MediaPipe requires RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            left_eye = landmarks.landmark[33]  # Left eye center
            right_eye = landmarks.landmark[133]  # Right eye center
            
            # normalize coordinates (0-1)
            h, w, _ = frame.shape
            LX = int(left_eye.x * w)
            LY = int(left_eye.y * h)
            RX = int(right_eye.x * w)
            RY = int(right_eye.y * h)
            
            return LX, LY, RX, RY

    # If no face or eyes are detected, return random values or None
    return np.random.random(), np.random.random(), np.random.random(), np.random.random()

# Path to the uploaded video
video_path = "/home/athul/Desktop/Dyslexia-Identification-Model/WhatsApp Video 2024-12-07 at 12.54.07.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Starting video stream...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    LX, LY, RX, RY = extract_eye_tracking_data(frame)

    sequence.append([LX, LY, RX, RY])

    if len(sequence) == time_steps:
        sequence_array = np.array(sequence).reshape(-1, num_features)
        sequence_array = scaler.fit_transform(sequence_array).reshape(1, time_steps, num_features)
        prediction = model.predict(sequence_array)
        dyslexia_prob = prediction[0][0]
        print(f"Dyslexia probability: {dyslexia_prob:.2f}")
        label = "Dyslexia Detected" if dyslexia_prob > 0.5 else "Normal"
        cv2.putText(frame, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        sequence = []

    cv2.imshow("Dyslexia Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
