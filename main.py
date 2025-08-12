# main.py
import json
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import piper
import re
import time  # <-- added

from faster_whisper import WhisperModel
from audio_test import record_seconds, resample_linear   

from memory import (
    load_facts, save_facts, append_history, load_recent_history,
    add_fact, forget_fact, summarize_facts, parse_memory_command
)

# Config 
OLLAMA_MODEL = "llama3"                 
OLLAMA_BASE  = "http://localhost:11434"
WHISPER_SIZE = "small"                  

# Piper voice model paths
PIPER_VOICE_ONNX = r".\voices\en_US-ryan-high.onnx"
PIPER_VOICE_CFG  = r".\voices\en_US-ryan-high.onnx.json"   
# Optional CLI fallback 
PIPER_EXE = shutil.which("piper.exe") or (os.path.abspath("./piper.exe") if os.path.exists("./piper.exe") else None)

SYSTEM_PROMPT = (
    "I am Kal. You are my voice assistant. Your name is Cyrus. Be concise, helpful, and keep context across turns. "
    "If the user says 'reset', acknowledge and start a fresh conversation. "
    "If the user says 'goodbye', say a short farewell."
)

# STT (CPU for stability) 
whisper = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
print("[whisper] using cpu/int8")

# Load memory
facts = load_facts()
print("[memory] loaded facts:", summarize_facts(facts) or "(none)")

# Piper TTS 
def load_voice():
    print(f"[piper] loading voice: {PIPER_VOICE_ONNX}")
    v = piper.PiperVoice.load(PIPER_VOICE_ONNX)  
    try:
        rate = int(getattr(v, "sample_rate", None) or v.config.sample_rate)
    except Exception:
        rate = 22050
    return v, rate

_tts, _rate = load_voice()

def _resample_float32(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resample float32 audio (supports mono or stereo)."""
    if src_rate == dst_rate:
        return x
    if x.size == 0:
        return x
    n_src = x.shape[0]
    n_dst = int(round(n_src * (dst_rate / src_rate)))
    src_idx = np.linspace(0, n_src - 1, num=n_src, endpoint=True)
    dst_idx = np.linspace(0, n_src - 1, num=n_dst, endpoint=True)
    if x.ndim == 1:
        y = np.interp(dst_idx, src_idx, x)
    else:
        y = np.column_stack([np.interp(dst_idx, src_idx, x[:, c]) for c in range(x.shape[1])])
    return y.astype(np.float32)

def _current_output_rate() -> int:
    """Return default output device's native sample rate (e.g., 48000 on gaming headsets)."""
    info = sd.query_devices(None, 'output')  
    return int(info['default_samplerate'])

def _split_sentences(text: str):
    # simple sentence splitter
    parts = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [p for p in parts if p]

def _synthesize_to_wav(text: str) -> tuple[str, int]:
    """
    Use piper-tts Python API to synthesize to a proper WAV.
    Returns (wav_path, sample_rate). May return (None, None) if nothing was produced.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()

   
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)       
        wf.setsampwidth(2)       
        wf.setframerate(_rate)   
        _tts.synthesize(text, wf)

    # Verify frames 
    try:
        with wave.open(tmp_path, "rb") as wf:
            frames = wf.getnframes()
            if frames == 0:
                return None, None
    except wave.Error:
        return None, None

    return tmp_path, _rate

def _synthesize_cli(text: str) -> tuple[str, int]:
    """Fallback: call piper.exe to synthesize a WAV. Returns (wav_path, sample_rate) or (None, None)."""
    if not PIPER_EXE:
        return None, None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()
    try:
        subprocess.run(
            [PIPER_EXE, "--model", PIPER_VOICE_ONNX, "--output_file", tmp_path],
            input=text.encode("utf-8"),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        with wave.open(tmp_path, "rb") as wf:
            rate = wf.getframerate()
            frames = wf.getnframes()
            if frames == 0:
                return None, None
        return tmp_path, rate
    except Exception:
        return None, None
    
def _fade_edges(x: np.ndarray, sr: int, ms: float = 5.0) -> np.ndarray:
    n = max(1, int(sr * ms / 1000.0))
    if x.size <= n: 
        return x
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x

def _noise_gate(x: np.ndarray, sr: int, win_ms: float = 20.0, thresh: float = 1e-3) -> np.ndarray:
    """Zero out frames whose short-term RMS is below thresh (mono or stereo)."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    hop = max(1, int(sr * win_ms / 1000.0))
    gate = np.ones(x.shape[0], dtype=bool)
    for start in range(0, x.shape[0], hop):
        end = min(x.shape[0], start + hop)
        rms = np.sqrt((x[start:end] ** 2).mean())
        if rms < thresh:
            gate[start:end] = False
    y = x.copy()
    y[~gate, :] = 0.0
    y = y.squeeze()
    return y

def speak(text: str, pause_sec: float = 0.28, length_scale: float = 1.05):
    """Sentence-by-sentence synthesis via Piper CLI, add pure silence between sentences, resample, and play."""
    text = (text or "").strip()
    if not text:
        return

    sentences = _split_sentences(text)
    if not sentences:
        return

    out_rate = _current_output_rate()
    chunks = []

    for sent in sentences:
        # synth one sentence 
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_path = tmp.name
        tmp.close()

        used_cli = False
        if PIPER_EXE:
            try:
                # NOTE: no sentence_silence here
                subprocess.run(
                    [
                        PIPER_EXE,
                        "--model", PIPER_VOICE_ONNX,
                        "--output_file", tmp_path,
                        "--length_scale", str(length_scale),
                    ],
                    input=sent.encode("utf-8"),
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                used_cli = True
            except Exception:
                used_cli = False

        if not used_cli:
            # Fallback: Python API, single sentence
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(_rate)
                _tts.synthesize(sent, wf)

       
        data, in_rate = sf.read(tmp_path, dtype="float32", always_2d=False)
        if data.size == 0:
            continue
        if in_rate != out_rate:
            data = _resample_float32(data, in_rate, out_rate)
        chunks.append(data)

        # Insert pure digital silence between sentences 
        if pause_sec > 0:
            chunks.append(np.zeros(int(pause_sec * out_rate), dtype=np.float32))

    if not chunks:
        print("[piper] got empty audio; skipping playback.")
        return

    audio = np.concatenate(chunks, axis=0)

    # tiny fade to avoid clicks at boundaries
    audio = _fade_edges(audio, out_rate, ms=6.0)

    sd.play(audio, out_rate)
    sd.wait()



# Ollama helpers 
def _parse_streaming(resp):
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
    lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)

def ask_ollama_with_history(messages, model=OLLAMA_MODEL):
    # 1) Try /api/chat non-stream
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=300,
        )
        if r.status_code == 200:
            return r.json()["message"]["content"].strip()
        if r.status_code != 404:
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

    # 2) Fallback to /api/generate
    prompt = _flatten_messages_for_prompt(messages)
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
    print("Transcribing...")
    if in_rate_hz != 16000:
        audio_int16 = resample_linear(audio_int16, in_rate_hz, 16000)
        in_rate_hz = 16000
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(in_rate_hz)
            wf.writeframes(audio_int16.tobytes())
        segments, _ = whisper.transcribe(tmp.name, beam_size=1)
    return "".join(s.text for s in segments).strip()

# Main loop 
if __name__ == "__main__":
    print("Jarvis is listening. Say 'goodbye' to exit, or 'reset' to clear memory.\n")

    # Seed messages with system prompt + memory
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if facts:
        messages.append({"role": "system", "content": "Known facts about Kal:\n" + summarize_facts(facts)})

    try:
        while True:
            # 1) Record speech (10 seconds)
            audio, in_rate, out_rate, in_idx, out_idx = record_seconds(10)

            # 2) STT
            text = transcribe_int16(audio, in_rate)
            if not text:
                print("(no speech detected, listening again…)\n")
                continue

            print(f"You said: {text}")

            # Log user turn
            append_history({"role": "user", "text": text, "ts": time.time()})

            # Local commands
            lower = text.lower().strip()
            if "reset" in lower:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                # also re-inject known facts after reset so they persist in context
                if facts:
                    messages.append({"role": "system", "content": "Known facts about Kal:\n" + summarize_facts(facts)})
                speak("Okay, I cleared our conversation.")
                print("Memory cleared. (Listening…)\n")
                continue
            if "goodbye" in lower or lower == "bye":
                speak("Goodbye!")
                print("Assistant: Goodbye! 👋")
                break

            # Memory commands: "remember ..." / "forget ..."
            cmd = parse_memory_command(text)
            if cmd:
                kind, payload = cmd
                if kind == "remember":
                    if add_fact(facts, payload):
                        save_facts(facts)
                        speak("Got it. I’ll remember that.")
                        print("[memory] remember:", payload)
                        messages.append({"role": "system", "content": f"New fact saved: {payload}"})
                    else:
                        speak("I already had that saved.")
                    continue
                elif kind == "forget":
                    if forget_fact(facts, payload):
                        save_facts(facts)
                        speak("Okay, I’ve forgotten that.")
                        print("[memory] forget:", payload)
                        messages.append({"role": "system", "content": f"Fact forgotten: {payload}"})
                    else:
                        speak("I couldn’t find that in memory.")
                    continue

            # Quick memory query
            if "what do you remember" in lower or "what do you know about me" in lower:
                summary = summarize_facts(facts) or "I don't have any saved facts yet."
                speak("Here's what I remember.")
                print(summary)
                messages.append({"role": "system", "content": "Reminder of known facts:\n" + summary})
                continue

            # 3) Append user turn
            messages.append({"role": "user", "content": text})

            # 4) Ask Ollama with history
            print("Asking Ollama…")
            reply = ask_ollama_with_history(messages)
            print("Assistant:", reply, "\n")

            # Log assistant turn
            append_history({"role": "assistant", "text": reply, "ts": time.time()})

            # 5) Append assistant turn
            messages.append({"role": "assistant", "content": reply})

            # 6) Speak reply
            speak(reply)

    except KeyboardInterrupt:
        print("\nExiting. Bye! 👋")
