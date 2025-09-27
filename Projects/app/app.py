import streamlit as st
from chatbot import MentalHealthChatbot
from emotion_detector import TextEmotionDetector, VoiceEmotionDetector
import tempfile
from streamlit_audiorecorder import audiorecorder
import os

chatbot = MentalHealthChatbot()
text_emotion = TextEmotionDetector()
voice_emotion = VoiceEmotionDetector()

st.title("🧠 Mental Health Chatbot with Emotion Detection")

# ==============================
# 📌 Text input
# ==============================
user_text = st.text_input("Type your message:")
if user_text:
    detected_emotion = text_emotion.predict_emotion(user_text)
    bot_reply = chatbot.get_response(user_text)
    st.write(f"**Detected Emotion (Text):** {detected_emotion}")
    st.write(f"**Bot:** {bot_reply}")

# ==============================
# 📌 Voice input
# ==============================
voice_file = st.file_uploader("Upload a voice clip (wav)", type=["wav"])
if voice_file:
    with open("temp.wav", "wb") as f:
        f.write(voice_file.read())
    try:
        emotion = voice_emotion.predict_emotion("temp.wav")
        st.write(f"**Detected Emotion (Voice):** {emotion}")
    except Exception as e:
        st.error(f"Error processing audio: {e}")


# ==============================
# 📌 Voice input (Record via mic)
# ==============================


st.subheader("🎙️ Record your voice")

audio = audiorecorder("Click to Record", "Recording...")

if len(audio) > 0:
    # Save recording to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio.tobytes())
        temp_path = temp_file.name

    try:
        emotion = voice_emotion.predict_emotion(temp_path)
        st.write(f"**Detected Emotion (Voice):** {emotion}")
    except Exception as e:
        st.error(f"Error processing audio: {e}")
