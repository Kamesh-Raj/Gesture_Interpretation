import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands()
face_mesh = mp_face.FaceMesh()

# Read labels
with open('labels.txt', 'r') as file:
    labels = [line.strip() for line in file]

# Output directory
output_dir = 'dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for label in labels:
    label_dir = os.path.join(output_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

# Capture video
cap = cv2.VideoCapture(0)

for label in labels:
    print(f"Press 's' to start recording for label '{label}', 'q' to quit")
    videos_recorded = 0

    while videos_recorded < 30:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('s'):
            print(f"Recording for label '{label}' started... ({videos_recorded+1}/30)")
            frames = []
            features = []

            start_time = time.time()
            while time.time() - start_time < 5:  # Record for 5 seconds
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)

                # Extract features
                results_hands = hands.process(rgb_frame)
                results_face = face_mesh.process(rgb_frame)

                frame_features = []
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        for lm in hand_landmarks.landmark:
                            frame_features.extend([lm.x, lm.y, lm.z])

                if results_face.multi_face_landmarks:
                    for face_landmarks in results_face.multi_face_landmarks:
                        for lm in face_landmarks.landmark:
                            frame_features.extend([lm.x, lm.y, lm.z])

                features.append(frame_features)

            features = np.array(features)
            np.save(os.path.join(label_dir, f'features_{videos_recorded}.npy'), features)
            print(f"Recording for label '{label}' saved. ({videos_recorded+1}/30)")
            videos_recorded += 1

        elif key & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Data collection completed.")
