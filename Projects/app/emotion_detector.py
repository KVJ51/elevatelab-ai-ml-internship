# ...existing code...
import os
import joblib
import librosa
import numpy as np


class TextEmotionDetector:
    def __init__(self):
        try:
            self.model = joblib.load("models/text_emotion_model.pkl")
            self.vectorizer = joblib.load("models/vectorizer.pkl")
        except Exception as e:
            print(f"Error loading model/vectorizer: {e}")
            raise

    def predict_emotion(self, text):
        try:
            X = self.vectorizer.transform([text])
            prediction = self.model.predict(X)[0]
            return prediction
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

class VoiceEmotionDetector:
    def __init__(self):
        try:
            self.model = joblib.load("models/voice_emotion_model.pkl")
        except Exception as e:
            print(f"Error loading voice model: {e}")
            raise

    def extract_features(self, file_path):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            y, sr = librosa.load(file_path, duration=3, offset=0.5)
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
            return mfccs.reshape(1, -1)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def predict_emotion(self, file_path):
        features = self.extract_features(file_path)
        if features is None:
            return None
        try:
            prediction = self.model.predict(features)[0]
            return prediction
        except Exception as e:
            print(f"Error during voice prediction: {e}")
            return None
# ...existing code...