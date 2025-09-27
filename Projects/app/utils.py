import joblib
import librosa
import numpy as np

# ==============================
# 📌 Model Loading Helpers
# ==============================
def load_model(model_path):
    """Load a trained model from a pickle file"""
    return joblib.load(model_path)

def load_vectorizer(vectorizer_path):
    """Load a trained vectorizer (for text preprocessing)"""
    return joblib.load(vectorizer_path)


# ==============================
# 📌 Audio Feature Extraction
# ==============================
def extract_audio_features(file_path, n_mfcc=40):
    """
    Extract MFCC features from an audio file.
    Parameters:
        file_path (str): Path to audio file (.wav)
        n_mfcc (int): Number of MFCCs to extract
    Returns:
        np.array: MFCC features (1D vector)
    """
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    return mfccs.reshape(1, -1)


# ==============================
# 📌 Text Preprocessing (if needed)
# ==============================
def preprocess_text(text, vectorizer):
    """
    Transform input text using a trained vectorizer.
    Parameters:
        text (str): User input text
        vectorizer: Fitted TfidfVectorizer
    Returns:
        vectorized form of input text
    """
    return vectorizer.transform([text])


# ==============================
# 📌 Emotion Label Mapping
# ==============================
emotion_labels = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}
