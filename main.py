import requests
from faster_whisper import WhisperModel
from audio_test import record_seconds, play

# Config
OLLAMA_MODEL = "llama3"

# Init Whisper (CPU for now — change to "cuda" later if GPU ready)
whisper = WhisperModel("small", device="cpu", compute_type="int8")

def transcribe(audio, in_rate):
    print("Transcribing...")
    # Whisper expects 16kHz — resample if needed
    import numpy as np
    if in_rate != 16000:
        from audio_test import resample_linear
        audio = resample_linear(audio, in_rate, 16000)
        in_rate = 16000
    # Save to temp wav
    import tempfile, wave
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp_wav.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(in_rate)
        wf.writeframes(audio.tobytes())
    segments, _ = whisper.transcribe(tmp_wav.name)
    text = "".join(s.text for s in segments).strip()
    return text

def ask_ollama(prompt):
    print("Asking Ollama...")
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        },
        stream=False
    )
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"].strip()

if __name__ == "__main__":
    audio, in_rate, out_rate, in_idx, out_idx = record_seconds(4)
    text = transcribe(audio, in_rate)
    print("You said:", text)

    if text:
        reply = ask_ollama(text)
        print("Ollama:", reply)
        # Convert reply to speech later with Piper — for now just play back your voice
        play(audio, in_rate, out_rate)
