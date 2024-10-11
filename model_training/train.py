import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load labels
with open('labels.txt', 'r') as file:
    labels = [line.strip() for line in file]

num_labels = len(labels)
num_frames = 100  # Set to a realistic value based on your data
fixed_length = 1548  # As defined in create_dataset.py

# Function to pad or truncate feature arrays
def pad_or_truncate(features, length):
    if len(features.shape) == 1:  # For 1D features
        if len(features) > length:
            return features[:length]
        else:
            return np.pad(features, (0, length - len(features)), 'constant')
    elif len(features.shape) == 2:  # For 2D features
        if features.shape[0] > length:
            return features[:length, :]
        else:
            return np.pad(features, ((0, length - features.shape[0]), (0, 0)), 'constant')
    return features

# Prepare dataset
X = []
y = []

for label in labels:
    label_dir = os.path.join('dataset', label)
    if not os.path.exists(label_dir):
        continue
    for filename in os.listdir(label_dir):
        if filename.endswith('.npy'):
            file_path = os.path.join(label_dir, filename)
            features = np.load(file_path)

            # Ensure the features array has the correct length
            features = pad_or_truncate(features, num_frames)
            X.append(features)
            y.append(labels.index(label))

X = np.array(X)
y = to_categorical(np.array(y), num_labels)

# Reshape X to match the expected input shape for the model
X = X.reshape(-1, num_frames, fixed_length)  # num_frames per video, each with fixed_length features

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(num_frames, fixed_length)),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_labels, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Save the model
model.save('gesture_model.h5')
