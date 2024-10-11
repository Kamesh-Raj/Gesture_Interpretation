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

# Fixed length for features per frame
fixed_length = 1548

def extract_hand_features(frame):
    """Extract hand features from a frame."""
    features = []
    results_hands = hands.process(frame)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
    return np.array(features)

def extract_face_features(frame):
    """Extract face features from a frame."""
    features = []
    results_face = face_mesh.process(frame)
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
    return np.array(features)

def process_frame(frame):
    """Extract combined features from a frame."""
    hand_features = extract_hand_features(frame)
    face_features = extract_face_features(frame)
    if hand_features is not None and face_features is not None:
        combined_features = np.concatenate((hand_features, face_features))
        # Pad or truncate to fixed length
        combined_features = combined_features[:fixed_length]  # truncate if longer
        combined_features = np.pad(combined_features, (0, max(0, fixed_length - len(combined_features))), 'constant')  # pad if shorter
        return combined_features
    return None

for label in labels:
    print(f"Press 's' to start recording for label '{label}', 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('s'):
            for videos_recorded in range(30):
                print(f"Recording for label '{label}' started... ({videos_recorded + 1}/30)")
                features_list = []

                start_time = time.time()
                while time.time() - start_time < 3:  # Record for 3 seconds
                    ret, frame = cap.read()
                    if not ret:
                        break
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Extract features
                    features = process_frame(rgb_frame)
                    if features is not None:
                        features_list.append(features)

                # Save features for each recorded video
                features_array = np.array(features_list)
                np.save(os.path.join(label_dir, f'{label}_{videos_recorded}.npy'), features_array)
                print(f"Recording for label '{label}' saved. ({videos_recorded + 1}/30)")

            break

        elif key & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Data collection completed.")
