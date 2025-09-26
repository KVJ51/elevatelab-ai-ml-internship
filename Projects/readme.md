🧠 AI-Powered Mental Health Chatbot with Emotion Detection

An AI-powered chatbot that combines Natural Language Processing (NLP) and Speech Emotion Recognition (SER) to provide supportive and empathetic conversations. The chatbot can detect emotions from text and voice recordings, and generate meaningful replies using DialoGPT.

🚀 Features

💬 Chatbot Interaction – Human-like responses using DialoGPT

😀 Text Emotion Detection – Classifies emotions from typed input

🎙️ Voice Emotion Detection – Detects emotions from recorded/uploaded voice clips

🖥️ Streamlit Web App – User-friendly interface for real-time interaction

📊 Trained Models – Custom-trained ML models for both text and voice emotions

🛠️ Tools & Technologies

Python

Streamlit (Frontend)

Hugging Face Transformers (DialoGPT chatbot)

Scikit-learn (ML models)

Librosa (Audio feature extraction)

Joblib (Model persistence)

Google Colab (Model training)

Datasets:

RAVDESS Dataset
 – Speech Emotion Recognition

Kaggle Emotion Text Dataset – Text Emotion Classification

⚡ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/mental_health_chatbot.git
cd mental_health_chatbot

2️⃣ Create Virtual Environment & Install Dependencies
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate   # On Linux/Mac

pip install -r requirements.txt

3️⃣ Run the App
streamlit run app/app.py

🎤 Using the App

Text Mode: Type your message → Detect emotion → Chatbot replies.

Voice Mode: Upload/record .wav file → Detect emotion → Chatbot replies.

📌 Future Enhancements

Multi-lingual emotion detection

More advanced deep learning models (CNNs, RNNs) for audio

Integration with professional mental health resources

Long-term emotion tracking dashboard

👩‍💻 Author

Varshini Janaki.K
B.Tech Information Technology – KGisl institute of Technology
📍 Coimbatore, India
