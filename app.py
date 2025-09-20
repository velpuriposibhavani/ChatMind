from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = Flask(__name__)

# Load dataset
data = pd.read_csv('chat_data.csv')

# Preprocess questions: lowercase & strip
questions = [q.lower().strip() for q in data['Question'].values]
answers = data['Answer'].values

# Vectorize questions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Chat history (memory)
chat_history = []

# Weather function
def get_weather(city):
    api_key = "ee4984622d21f774c477c72cdf243541"  # ✅ your real API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url).json()
    if response.get("main"):
        temp = response["main"]["temp"]
        condition = response["weather"][0]["description"]
        return f"🌤 Weather in {city.capitalize()}: {temp}°C, {condition}"
    else:
        return "❌ City not found."

# Get bot response
def get_response(user_input):
    user_input_clean = user_input.lower().strip()

    # 1️⃣ Exact match check
    if user_input_clean in questions:
        index = questions.index(user_input_clean)
        bot_reply = answers[index]
        chat_history.append(f"You: {user_input}")
        chat_history.append(f"Bot: {bot_reply}")
        return bot_reply

    # 2️⃣ Check AIML variations manually
    if "aiml" in user_input_clean:
        bot_reply = "AIML stands for Artificial Intelligence and Machine Learning. AI means Artificial Intelligence, ML means Machine Learning."
        chat_history.append(f"You: {user_input}")
        chat_history.append(f"Bot: {bot_reply}")
        return bot_reply

    # 3️⃣ Check if weather query
    if "weather" in user_input_clean:
        words = user_input_clean.split()
        city = words[-1]  # take last word as city
        bot_reply = get_weather(city)
        chat_history.append(f"You: {user_input}")
        chat_history.append(f"Bot: {bot_reply}")
        return bot_reply

    # 4️⃣ Normal Q&A from dataset using TF-IDF similarity
    user_vec = vectorizer.transform([user_input_clean])
    similarity = cosine_similarity(user_vec, X)
    index = similarity.argmax()
    bot_reply = answers[index]

    # Save to history
    chat_history.append(f"You: {user_input}")
    chat_history.append(f"Bot: {bot_reply}")
    return bot_reply

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return get_response(userText)

if __name__ == "__main__":
    app.run(debug=True)
