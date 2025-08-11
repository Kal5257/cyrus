import json
import tempfile
import wave
import requests
from faster_whisper import WhisperModel
from audio_test import record_seconds, play, resample_linear

# Config 
OLLAMA_MODEL = "llama3"          
OLLAMA_BASE  = "http://localhost:11434"
WHISPER_SIZE = "small"           


whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")

# Ollama helpers 
def _parse_streaming(resp):
    """Parse NDJSON streaming responses from Ollama."""
    text_parts = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # /api/chat
        if "message" in obj and obj["message"] and "content" in obj["message"]:
            text_parts.append(obj["message"]["content"])
        # /api/generate
        elif "response" in obj:
            text_parts.append(obj["response"])
        if obj.get("done"):
            break
    return "".join(text_parts).strip()

def ask_ollama(prompt, model=OLLAMA_MODEL):
    """Call Ollama, preferring /api/chat (non-stream), then falling back to streaming or /api/generate."""
    # 1) Try chat endpoint (non-stream)
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=300,
        )
        if r.status_code == 200:
            data = r.json()
            return data["message"]["content"].strip()
        if r.status_code != 404:
            # Some servers ignore stream=False and still stream
            r = requests.post(
                f"{OLLAMA_BASE}/api/chat",
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True},
                timeout=300,
                stream=True,
            )
            r.raise_for_status()
            return _parse_streaming(r)
    except requests.RequestException:
        pass

    # 2) Fallback to /api/generate
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=300,
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("response", "").strip()
    except requests.JSONDecodeError:
        # Server streamed anyway; fall through to stream mode
        pass

    # 3) /api/generate 
    r = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True},
        timeout=300,
        stream=True,
    )
    r.raise_for_status()
    return _parse_streaming(r)

# STT helper
def transcribe_int16(audio_int16, in_rate_hz: int) -> str:
    """Transcribe mono int16 audio at in_rate_hz with Whisper."""
    print("Transcribing...")
 
    if in_rate_hz != 16000:
        audio_int16 = resample_linear(audio_int16, in_rate_hz, 16000)
        in_rate_hz = 16000

    # Write temp wav
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) 
            wf.setframerate(in_rate_hz)
            wf.writeframes(audio_int16.tobytes())
        segments, _ = whisper.transcribe(tmp.name)
    text = "".join(s.text for s in segments).strip()
    return text

# Main -
if __name__ == "__main__":
    # Record
    audio, in_rate, out_rate, in_idx, out_idx = record_seconds(4)

    # STT
    text = transcribe_int16(audio, in_rate)
    print("You said:", text or "[no speech detected]")

    if not text:
        exit(0)

    # LLM
    print("Asking Ollama...")
    reply = ask_ollama(text)
    print("Ollama:", reply)

  
