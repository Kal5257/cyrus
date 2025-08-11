import json
import tempfile
import wave
import requests
from faster_whisper import WhisperModel
from audio_test import record_seconds, resample_linear


# Config 
OLLAMA_MODEL = "llama3"         
OLLAMA_BASE  = "http://localhost:11434"
WHISPER_SIZE = "small"          

SYSTEM_PROMPT = (
    "You are Kal's private voice assistant. Be concise, helpful, and keep context across turns. "
    "If the user says 'reset', acknowledge and start a fresh conversation. "
    "If the user says 'goodbye', say a short farewell."
)

# Init Whisper (CPU for now; switch to device='cuda' when GPU is ready)
whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

# ---------- Ollama helpers ----------
def _parse_streaming(resp):
    """Parse NDJSON streaming from Ollama."""
    parts = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "message" in obj and obj["message"] and "content" in obj["message"]:
            parts.append(obj["message"]["content"])
        elif "response" in obj:
            parts.append(obj["response"])
        if obj.get("done"):
            break
    return "".join(parts).strip()

def _flatten_messages_for_prompt(messages):
    """Fallback prompt for /api/generate if /api/chat isn't available."""
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)

def ask_ollama_with_history(messages, model=OLLAMA_MODEL):
    """Prefer /api/chat; fall back to /api/generate (stream/non-stream)."""
    # 1) Try chat non-stream
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=300,
        )
        if r.status_code == 200:
            return r.json()["message"]["content"].strip()
        if r.status_code != 404:
            # Some servers may still stream
            r = requests.post(
                f"{OLLAMA_BASE}/api/chat",
                json={"model": model, "messages": messages, "stream": True},
                timeout=300,
                stream=True,
            )
            r.raise_for_status()
            return _parse_streaming(r)
    except requests.RequestException:
        pass

    # 2) Fallback: /api/generate
    prompt = _flatten_messages_for_prompt(messages)
    # try non-stream
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=300,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except requests.JSONDecodeError:
        pass

    # stream parse
    r = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True},
        timeout=300,
        stream=True,
    )
    r.raise_for_status()
    return _parse_streaming(r)

# ---------- STT ----------
def transcribe_int16(audio_int16, in_rate_hz: int) -> str:
    """Transcribe mono int16 audio at in_rate_hz with Whisper."""
    print("Transcribing...")
    # Whisper expects 16kHz; resample if needed
    if in_rate_hz != 16000:
        audio_int16 = resample_linear(audio_int16, in_rate_hz, 16000)
        in_rate_hz = 16000

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(in_rate_hz)
            wf.writeframes(audio_int16.tobytes())
        segments, _ = whisper.transcribe(tmp.name)

    text = "".join(s.text for s in segments).strip()
    return text

# ---------- Main loop ----------
if __name__ == "__main__":
    print("Jarvis is listening. Say 'goodbye' to exit, or 'reset' to clear memory.\n")

    # Running conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while True:
            # 1) Record speech
            audio, in_rate, out_rate, in_idx, out_idx = record_seconds(10)

            # 2) STT
            text = transcribe_int16(audio, in_rate)
            if not text:
                print("(no speech detected, listening again…)\n")
                continue

            print(f"You said: {text}")

            # Handle local commands
            lower = text.lower().strip()
            if "reset" in lower:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                print("Memory cleared. (Listening…)\n")
                # Optionally acknowledge with TTS later
                continue
            if "goodbye" in lower or "bye" == lower:
                print("Assistant: Goodbye! 👋")
                break

            # 3) Append user turn
            messages.append({"role": "user", "content": text})

            # 4) Ask Ollama with full history
            print("Asking Ollama…")
            reply = ask_ollama_with_history(messages)
            print("Assistant:", reply, "\n")

            # 5) Append assistant turn
            messages.append({"role": "assistant", "content": reply})

            # 6) (Optional) speak reply with TTS here (we’ll add Piper next)

    except KeyboardInterrupt:
        print("\nExiting. Bye! 👋")

  
