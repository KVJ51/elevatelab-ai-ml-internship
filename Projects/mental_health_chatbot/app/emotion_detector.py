import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

class TextEmotionDetector:
    def __init__(self):
        self.model = joblib.load("models/text_emotion_model.pkl")
        self.vectorizer = joblib.load("models/vectorizer.pkl")

    def predict_emotion(self, text):
        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]
        return prediction

import librosa
import numpy as np
import joblib

class VoiceEmotionDetector:
    def __init__(self):
        self.model = joblib.load("models/voice_emotion_model.pkl")

    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfccs.reshape(1, -1)

    def predict_emotion(self, file_path):
        features = self.extract_features(file_path)
        prediction = self.model.predict(features)[0]
        return prediction
