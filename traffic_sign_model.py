
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class TrafficSignModel:
    def __init__(self):
        self.model = self.create_model()
    
    def create_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def predict_sign(self, frame):
        resized_frame = cv2.resize(frame, (32, 32)) / 255.0
        reshaped_frame = np.expand_dims(resized_frame, axis=0)
        prediction = self.model.predict(reshaped_frame)
        return np.argmax(prediction)
