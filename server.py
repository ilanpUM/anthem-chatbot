import os
import secrets

from flask import Flask, Response, jsonify, request

from chatbot import AnthemChatbot, Step  # also loads .env

from elevenlabs.client import ElevenLabs

app = Flask(__name__, static_folder="static")
app.secret_key = secrets.token_hex(32)

el_client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
# Matilda — Knowledgeable, Professional, middle-aged female
VOICE_ID = "XrExE9yKIg1WjnnlVkGX"
TTS_MODEL = "eleven_turbo_v2_5"


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/start", methods=["POST"])
def start():
    """Return the greeting and initial step name — no server-side session needed."""
    bot = AnthemChatbot()
    return jsonify({"step": bot.step.name, "message": bot.greeting(), "ended": False})


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Stateless: client sends the current step name with each message.
    Server reconstructs the bot at that step, processes the input,
    and returns the new step name so the client can track state.
    """
    data = request.get_json() or {}
    step_name  = data.get("step", "S1")
    user_input = data.get("message", "").strip()

    try:
        current_step = Step[step_name]
    except KeyError:
        return jsonify({"error": f"Unknown step: {step_name}"}), 400

    bot = AnthemChatbot()
    bot.step = current_step

    reply = bot.respond(user_input)
    return jsonify({"step": bot.step.name, "message": reply, "ended": bot.ended})


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
