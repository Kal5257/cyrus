import sys, types

# sounddevice stub
sd = types.SimpleNamespace()
def _query_devices(index=None, kind=None):
    # minimal shape to satisfy main._current_output_rate()
    if kind == 'output':
        return {'default_samplerate': 48000}
    return {'default_samplerate': 48000}
sd.query_devices = _query_devices
sd.play = lambda *a, **k: None
sd.wait = lambda *a, **k: None
sd.rec  = lambda *a, **k: None
sd.default = types.SimpleNamespace(device=None, channels=None)
sys.modules.setdefault("sounddevice", sd)

# soundfile stub
sf = types.SimpleNamespace()
sf.read = lambda *a, **k: ([0.0], 48000)  # (data, samplerate)
sys.modules.setdefault("soundfile", sf)

# piper stub
p = types.SimpleNamespace()
class FakeVoice:
    config = types.SimpleNamespace(sample_rate=22050)
    def synthesize(self, text, wf):
        wf.writeframes(b"\x00\x00")
p.PiperVoice = types.SimpleNamespace(load=lambda *args, **kwargs: FakeVoice())
sys.modules.setdefault("piper", p)

# faster_whisper stub
fw = types.SimpleNamespace()
class FakeWhisperModel:
    def __init__(self, *a, **k): pass
    def transcribe(self, *a, **k):
        class Seg: 
            def __init__(self, t): self.text = t
        return [Seg("hello")], None
fw.WhisperModel = FakeWhisperModel
sys.modules.setdefault("faster_whisper", fw)

# --- now it's safe to import main ---
import json
import main

def test_split_sentences_basic():
    text = "Hello. How are you? I'm fine!"
    parts = main._split_sentences(text)
    assert parts == ["Hello.", "How are you?", "I'm fine!"]

def test_flatten_messages_for_prompt():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hey"},
        {"role": "user", "content": "Bye"},
    ]
    out = main._flatten_messages_for_prompt(messages)
    assert out.splitlines()[-1] == "ASSISTANT:"
    assert "SYSTEM: SYS" in out
    assert "USER: Hi" in out
    assert "ASSISTANT: Hey" in out
    assert "USER: Bye" in out

def test_parse_streaming_collects_chunks(monkeypatch):
    chunks = [
        json.dumps({"message": {"content": "Hello "}}),
        json.dumps({"message": {"content": "Kal!"}}),
        json.dumps({"done": True})
    ]
    resp = types.SimpleNamespace(iter_lines=lambda decode_unicode: chunks)
    combined = main._parse_streaming(resp)
    assert combined == "Hello Kal!"

def test_ask_ollama_with_history_uses_chat_nonstream(monkeypatch):
    class Resp:
        status_code = 200
        def json(self):
            return {"message": {"content": "chat-ok"}}
    def fake_post(url, json=None, timeout=None, stream=False):
        return Resp()
    monkeypatch.setattr(main.requests, "post", fake_post)
    result = main.ask_ollama_with_history([{"role": "user", "content": "Hi"}], model="anything")
    assert result == "chat-ok"

def test_ask_ollama_with_history_fallback_generate(monkeypatch):
    class RespGen:
        status_code = 200
        def json(self):
            return {"response": "gen-ok"}
    def fake_post(url, json=None, timeout=None, stream=False):
        if url.endswith("/api/chat"):
            raise main.requests.RequestException("fail chat")
        if url.endswith("/api/generate"):
            return RespGen()
        raise AssertionError("Unexpected URL " + url)
    monkeypatch.setattr(main.requests, "post", fake_post)
    result = main.ask_ollama_with_history([{"role": "user", "content": "Hi"}], model="anything")
    assert result == "gen-ok"

def test_speak_noop_when_piper_unavailable(monkeypatch):
    # Force no-audio path; should not raise
    monkeypatch.setattr(main, "HAS_PIPER", False)
    main.speak("Hello, world.")
