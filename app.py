from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from googletrans import Translator
from collections import deque
import nltk
import pyttsx3
import threading
import time

# Initialize Flask app
app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('gesture_model.h5')
translator = Translator()

# Initialize NLTK
nltk.download('punkt')

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands()
face_mesh = mp_face.FaceMesh()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Buffer to store frames for 3 seconds
frame_buffer = deque(maxlen=90)  # Assuming 30 FPS
recognition_active = False
recognized_gesture = ''
translated_gesture = ''
selected_language = 'en'  # Default language

@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/set_language', methods=['POST'])
def set_language():
    """Set the selected language for translation."""
    global selected_language
    selected_language = request.form['language']
    return jsonify({'message': 'Language updated.'})

def generate_frames():
    """Generate video frames for streaming."""
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(rgb_frame)
        results_face = face_mesh.process(rgb_frame)

        # Draw landmarks on frame
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face.FACEMESH_CONTOURS)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Provide video feed for streaming."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognize', methods=['POST'])
def recognize():
    """Recognize the gesture from video feed."""
    global recognition_active
    if recognition_active:
        return jsonify({
            'recognized': recognized_gesture,
            'translated': translated_gesture
        })
    return jsonify({'recognized': '', 'translated': ''})

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    """Start gesture recognition."""
    global recognition_active
    recognition_active = True
    threading.Thread(target=gesture_recognition_loop).start()  # Start recognition loop in a separate thread
    return jsonify({'message': 'Recognition started.'})

@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    """Stop gesture recognition."""
    global recognition_active
    recognition_active = False
    return jsonify({'message': 'Recognition stopped.'})

@app.route('/clear', methods=['POST'])
def clear():
    """Clear the recognized gesture and translated gesture fields."""
    global recognized_gesture, translated_gesture
    recognized_gesture = ''
    translated_gesture = ''
    return jsonify({'recognized': recognized_gesture, 'translated': translated_gesture})

@app.route('/backspace', methods=['POST'])
def backspace():
    """Remove the last character from the recognized gesture."""
    global recognized_gesture
    recognized_gesture = recognized_gesture[:-1]
    return jsonify({'recognized': recognized_gesture})

def gesture_recognition_loop():
    """Continuously recognize gestures and update recognized and translated gestures."""
    global recognized_gesture, translated_gesture

    while recognition_active:
        recognized_gesture = recognize_gesture()
        if recognized_gesture:  # Only translate if there is a recognized gesture
            # Translate recognized gesture
            translated_gesture = translator.translate(recognized_gesture, dest=selected_language).text
            speak(translated_gesture)  # Speak the translated gesture
            
            # Append the recognized gesture to output.txt
            with open('output.txt', 'a') as f:
                f.write(recognized_gesture + '\n')  # Append the recognized gesture with a newline

        time.sleep(1)  # Adjust the sleep time as needed to control recognition frequency

def recognize_gesture():
    """Recognize the gesture using the trained model."""
    frame_buffer.clear()
    for _ in range(90):  # Capture frames for 3 seconds
        success, frame = cap.read()
        if not success:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(rgb_frame)

    # Extract features from frames
    features = extract_features(frame_buffer)

    if features is None:
        return None

    # Check if features array has the expected shape
    if features.shape != (100, 1548):  # Adjust to your model's expected shape
        print(f"Unexpected features shape: {features.shape}, cannot reshape to (1, 100, 1548)")
        return None

    # Predict gesture
    features = np.expand_dims(features, axis=0)  # Add batch dimension: shape becomes (1, 100, 1548)
    predictions = model.predict(features)  # Adjust to your model's expected shape
    gesture_index = np.argmax(predictions, axis=1)[0]

    gesture_label = map_index_to_label(gesture_index)

    return gesture_label

def extract_features(frames):
    """Extract features from the frames using MediaPipe."""
    feature_list = []
    
    for frame in frames:
        hand_features = extract_hand_features(frame)  # Shape (n_hand_features,)
        face_features = extract_face_features(frame)  # Shape (n_face_features,)

        if hand_features is not None and face_features is not None:
            combined_features = np.concatenate((hand_features, face_features))  # Combine features
            feature_list.append(combined_features)
    
    # Ensure we have 100 frames for consistent input shape
    while len(feature_list) < 100:
        feature_list.append(np.zeros(1548))  # Fill with zeros if not enough frames

    if len(feature_list) > 100:
        feature_list = feature_list[:100]  # Truncate if too many frames
    
    return np.array(feature_list)  # Shape will be (100, 1548)

def extract_hand_features(frame):
    """Extract hand landmarks as features."""
    hand_features = []
    results_hands = hands.process(frame)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                hand_features.extend([lm.x, lm.y, lm.z])
    else:
        hand_features = [0] * (21 * 3)  # Append zeros if no hands detected

    # Ensure consistent size
    if len(hand_features) < 21 * 3:
        hand_features.extend([0] * (21 * 3 - len(hand_features)))  # Pad with zeros
    elif len(hand_features) > 21 * 3:
        hand_features = hand_features[:21 * 3]  # Truncate if too many features

    return np.array(hand_features)

def extract_face_features(frame):
    """Extract face landmarks as features."""
    face_features = []
    results_face = face_mesh.process(frame)
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                face_features.extend([lm.x, lm.y, lm.z])
    else:
        face_features = [0] * (468 * 3)  # Append zeros if no face detected

    # Ensure consistent size
    if len(face_features) < 468 * 3:
        face_features.extend([0] * (468 * 3 - len(face_features)))  # Pad with zeros
    elif len(face_features) > 468 * 3:
        face_features = face_features[:468 * 3]  # Truncate if too many features

    return np.array(face_features)


def map_index_to_label(index):
    """Map the index of the gesture to the corresponding label."""
    with open('labels.txt', 'r') as file:
        labels = [line.strip() for line in file]
    return labels[index]

def speak(text):
    """Convert text to speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
