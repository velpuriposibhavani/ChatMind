
import os
import requests
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia

# Optional backend TTS/voice (only if you want server-side voice)
# import pyttsx3
# import speech_recognition as sr

app = Flask(__name__)

# ---------- Load dataset ----------
DATA_PATH = "chat_data.csv"
data = pd.read_csv(DATA_PATH)
questions = data['Question'].astype(str).values
answers = data['Answer'].astype(str).values

# ---------- Vectorizer ----------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# ---------- Memory ----------
chat_history = []   # stores dicts: {"speaker":"You"/"ChatMind", "msg": "..."}
MAX_CONTEXT = 3     # how many previous user messages to include for context

# ---------- Helpers: domain-specific functions ----------
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")  # set this in your env if using weather

def get_weather(city):
    """Return a short weather string using OpenWeatherMap API. Requires OPENWEATHER_API_KEY env variable."""
    if not OPENWEATHER_API_KEY:
        return "Weather feature requires OpenWeather API key. Set OPENWEATHER_API_KEY environment variable."
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        r = requests.get(url, params=params, timeout=6)
        r.raise_for_status()
        j = r.json()
        desc = j['weather'][0]['description']
        temp = j['main']['temp']
        return f"{city.title()} weather: {desc}, {temp}°C"
    except Exception as e:
        return "Sorry, couldn't fetch weather right now."

def get_wikipedia_summary(query):
    try:
        s = wikipedia.summary(query, sentences=2, auto_suggest=True, redirect=True)
        return s
    except Exception as e:
        return "Sorry, I couldn't find info on that topic."

# ---------- Main response function ----------
def get_response(user_input):
    user_input = str(user_input).strip()
    # Build context (last few user messages) to enrich query
    recent_user_msgs = [h['msg'] for h in chat_history if h['speaker']=="You"][-MAX_CONTEXT:]
    context_prefix = " ".join(recent_user_msgs + [user_input]) if recent_user_msgs else user_input

    # Domain rules: check weather intent
    low = user_input.lower()
    # very simple heuristics for weather
    if "weather" in low or "temperature" in low or "rain" in low:
        # try to extract city (naive: last word or after 'in')
        city = None
        if " in " in low:
            city = low.split(" in ")[-1].strip().split("?")[0]
        else:
            parts = low.split()
            if len(parts) >= 1:
                city = parts[-1]
        if city:
            response = get_weather(city)
        else:
            response = "Which city? (e.g., 'Weather in Hyderabad')"
    # wikipedia / who is / what is
    elif low.startswith("who is") or low.startswith("what is") or "wikipedia" in low or "tell me about" in low:
        # remove trigger words
        q = user_input.replace("who is", "").replace("what is", "").replace("tell me about", "").replace("wikipedia", "").strip()
        if not q:
            response = "Please tell me the topic name (e.g., 'Who is APJ Abdul Kalam')."
        else:
            response = get_wikipedia_summary(q)
    else:
        # fallback to TF-IDF similarity on dataset (using context prefix)
        user_vec = vectorizer.transform([context_prefix])
        similarity = cosine_similarity(user_vec, X)
        idx = int(similarity.argmax())
        score = similarity.max()
        # If similarity is low, fallback to a safe response
        if score < 0.1:
            response = "Sorry, I don't know that yet. You can ask me about weather (e.g., 'Weather in Hyderabad') or ask 'Who is ...' for Wikipedia info."
        else:
            response = answers[idx]

    # Save conversation in memory
    chat_history.append({"speaker": "You", "msg": user_input})
    chat_history.append({"speaker": "ChatMind", "msg": response})

    return response

# ---------- Flask routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg', '')
    return get_response(userText)

@app.route("/history")
def get_history():
    return jsonify(chat_history)

# Optional route to clear history
@app.route("/clear_history", methods=['POST'])
def clear_history():
    chat_history.clear()
    return jsonify({"ok": True})

if __name__ == "__main__":
    # for local dev, set host=127.0.0.1 and debug True
    app.run(debug=True)
