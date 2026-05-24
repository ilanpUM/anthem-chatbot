import secrets
import uuid

from flask import Flask, jsonify, request

from chatbot import AnthemChatbot

app = Flask(__name__, static_folder="static")
app.secret_key = secrets.token_hex(32)

# In-memory store: session_id -> AnthemChatbot
_sessions: dict[str, AnthemChatbot] = {}


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


if __name__ == "__main__":
    app.run(debug=True, port=8080)
