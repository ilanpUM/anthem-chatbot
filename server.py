import os
import secrets
import uuid

from flask import Flask, Response, jsonify, request

from chatbot import AnthemChatbot  # also loads .env

from elevenlabs.client import ElevenLabs

app = Flask(__name__, static_folder="static")
app.secret_key = secrets.token_hex(32)

# In-memory store: session_id -> AnthemChatbot
_sessions: dict[str, AnthemChatbot] = {}

el_client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
# Matilda — Knowledgeable, Professional, middle-aged female
VOICE_ID = "XrExE9yKIg1WjnnlVkGX"
TTS_MODEL = "eleven_turbo_v2_5"


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/start", methods=["POST"])
def start():
    session_id = str(uuid.uuid4())
    bot = AnthemChatbot()
    _sessions[session_id] = bot
    return jsonify({"session_id": session_id, "message": bot.greeting(), "ended": False})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    session_id = data.get("session_id", "")
    user_input = data.get("message", "").strip()

    bot = _sessions.get(session_id)
    if not bot:
        return jsonify({"error": "Session not found. Please start a new call."}), 404

    if bot.ended:
        return jsonify({"message": "", "ended": True})

    reply = bot.respond(user_input)
    return jsonify({"message": reply, "ended": bot.ended})


@app.route("/api/tts", methods=["POST"])
def tts():
    text = (request.get_json() or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    audio_stream = el_client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id=TTS_MODEL,
        output_format="mp3_44100_128",
    )
    audio_bytes = b"".join(audio_stream)
    return Response(audio_bytes, mimetype="audio/mpeg")


if __name__ == "__main__":
    app.run(debug=True, port=8080)
