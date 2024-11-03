import cv2
import numpy as np
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

def extract_eye_tracking_data(frame):
    return np.random.random(), np.random.random(), np.random.random(), np.random.random()

# Start video capture
cap = cv2.VideoCapture(0)

print("Starting video stream...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract eye-tracking data
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

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
