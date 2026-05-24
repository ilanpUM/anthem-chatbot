import asyncio
import json
import os

import websockets.legacy.client as ws_legacy
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from deepgram import AsyncDeepgramClient
from deepgram.listen.v1.types import ListenV1Results
from elevenlabs.client import ElevenLabs

from chatbot import AnthemChatbot, Step  # also loads .env

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

el_client   = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
VOICE_ID    = "XrExE9yKIg1WjnnlVkGX"   # Matilda — professional, middle-aged female
TTS_MODEL   = "eleven_turbo_v2_5"

DG_API_KEY  = os.environ["DEEPGRAM_API_KEY"]
AAI_API_KEY = os.environ["ASSEMBLYAI_API_KEY"]

# ---------------------------------------------------------------------------
# Static / HTML
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# Chatbot REST API  (stateless — step travels with every request)
# ---------------------------------------------------------------------------
@app.post("/api/start")
async def start():
    bot = AnthemChatbot()
    return JSONResponse({"step": bot.step.name, "message": bot.greeting(), "ended": False})


@app.post("/api/chat")
async def chat(request: Request):
    data       = await request.json()
    step_name  = data.get("step", "S1")
    user_input = data.get("message", "").strip()
    try:
        current_step = Step[step_name]
    except KeyError:
        return JSONResponse({"error": f"Unknown step: {step_name}"}, status_code=400)
    bot       = AnthemChatbot()
    bot.step  = current_step
    reply     = bot.respond(user_input)
    return JSONResponse({"step": bot.step.name, "message": reply, "ended": bot.ended})


# ---------------------------------------------------------------------------
# ElevenLabs TTS
# ---------------------------------------------------------------------------
@app.post("/api/tts")
async def tts(request: Request):
    data = await request.json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return JSONResponse({"error": "No text"}, status_code=400)
    audio_stream = el_client.text_to_speech.convert(
        voice_id=VOICE_ID, text=text,
        model_id=TTS_MODEL, output_format="mp3_44100_128",
    )
    return Response(content=b"".join(audio_stream), media_type="audio/mpeg")


# ---------------------------------------------------------------------------
# STT WebSocket  (Silero VAD gated PCM → Deepgram + AssemblyAI fan-out)
# ---------------------------------------------------------------------------
AAI_WS_URL = (
    "wss://streaming.assemblyai.com/v3/ws"
    "?sample_rate=16000"
    "&encoding=pcm_s16le"
    "&end_utterance_silence_threshold=700"
)


@app.websocket("/ws/stt")
async def stt_ws(websocket: WebSocket):
    await websocket.accept()

    dg_client = AsyncDeepgramClient(api_key=DG_API_KEY)

    async with dg_client.listen.v1.connect(
        model="nova-2",
        encoding="linear16",
        sample_rate=16000,
        interim_results=True,
        smart_format=True,
        endpointing=300,
    ) as dg_socket:
        async with ws_legacy.connect(
            AAI_WS_URL, extra_headers={"Authorization": AAI_API_KEY}
        ) as aai_socket:

            # ── forward browser audio to both services ──────────────────
            async def forward_audio():
                try:
                    async for pcm in websocket.iter_bytes():
                        await asyncio.gather(
                            dg_socket.send_media(pcm),
                            aai_socket.send(pcm),
                            return_exceptions=True,
                        )
                except WebSocketDisconnect:
                    pass
                finally:
                    try:
                        await dg_socket.send_finalize()
                    except Exception:
                        pass
                    try:
                        await aai_socket.close()
                    except Exception:
                        pass

            # ── Deepgram → interim / final transcripts ──────────────────
            async def recv_deepgram():
                try:
                    async for msg in dg_socket:
                        if not isinstance(msg, ListenV1Results):
                            continue
                        alt = msg.channel.alternatives[0]
                        if not alt.transcript:
                            continue
                        event = "final" if msg.is_final else "interim"
                        try:
                            await websocket.send_json({"type": event, "text": alt.transcript})
                        except Exception:
                            return
                except Exception:
                    pass

            # ── AssemblyAI → semantic utterance end ─────────────────────
            async def recv_assemblyai():
                try:
                    async for raw in aai_socket:
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        if msg.get("message_type") == "FinalTranscript" and msg.get("text"):
                            try:
                                await websocket.send_json(
                                    {"type": "utterance_end", "text": msg["text"]}
                                )
                            except Exception:
                                return
                except Exception:
                    pass

            # forward_audio drives the lifecycle — recv tasks run until cancelled
            fwd = asyncio.create_task(forward_audio())
            dg_recv  = asyncio.create_task(recv_deepgram())
            aai_recv = asyncio.create_task(recv_assemblyai())
            try:
                await fwd
            finally:
                dg_recv.cancel()
                aai_recv.cancel()
                await asyncio.gather(dg_recv, aai_recv, return_exceptions=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
