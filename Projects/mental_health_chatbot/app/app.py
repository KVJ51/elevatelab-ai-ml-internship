import streamlit as st
from chatbot import MentalHealthChatbot
from emotion_detector import TextEmotionDetector, VoiceEmotionDetector

chatbot = MentalHealthChatbot()
text_emotion = TextEmotionDetector()
voice_emotion = VoiceEmotionDetector()

st.title("🧠 Mental Health Chatbot with Emotion Detection")

# Text input
user_text = st.text_input("Type your message:")
if user_text:
    detected_emotion = text_emotion.predict_emotion(user_text)
    bot_reply = chatbot.get_response(user_text)
    st.write(f"**Detected Emotion (Text):** {detected_emotion}")
    st.write(f"**Bot:** {bot_reply}")

# Voice input (upload file)
voice_file = st.file_uploader("Upload a voice clip (wav)", type=["wav"])
if voice_file:
    with open("temp.wav", "wb") as f:
        f.write(voice_file.read())
    emotion = voice_emotion.predict_emotion("temp.wav")
    st.write(f"**Detected Emotion (Voice):** {emotion}")
