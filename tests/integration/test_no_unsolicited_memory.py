# tests/integration/test_no_unsolicited_memory.py
# Stub heavy deps BEFORE importing main
import sys, types

# sounddevice stub
sd = types.SimpleNamespace()
sd.query_devices = lambda index=None, kind=None: {'default_samplerate': 48000}
sd.play = lambda *a, **k: None
sd.wait = lambda *a, **k: None
sd.rec  = lambda *a, **k: None
sd.default = types.SimpleNamespace(device=None, channels=None)
sys.modules.setdefault("sounddevice", sd)

# soundfile stub
sf = types.SimpleNamespace()
sf.read = lambda *a, **k: ([0.0], 48000)
sys.modules.setdefault("soundfile", sf)

# piper stub
p = types.SimpleNamespace()
class FakeVoice:
    config = types.SimpleNamespace(sample_rate=22050)
    def synthesize(self, text, wf): wf.writeframes(b"\x00\x00")
p.PiperVoice = types.SimpleNamespace(load=lambda *a, **k: FakeVoice())
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

# Now safe to import main
import re
import main

def test_sanitize_unsolicited_memory_reply_strips_facts():
    # Simulate stored facts
    fake_facts = {
        "profile": {"name": "Kal", "pronouns": "she/her"},
        "preferences": {},
        "facts": [
            {"text": "pizza", "added_at": 1},
            {"text": "black", "added_at": 2},
        ]
    }

    # Model overshares
    raw = "Hello Kal! I remember your favorite color is black and you love pizza. How can I help?"
    cleaned = main.sanitize_unsolicited_memory_reply(raw, user_asked_for_memory=False, facts=fake_facts)

    # Must remove leaks
    low = cleaned.lower()
    assert "remember" not in low
    assert "pizza" not in low
    assert "black" not in low

    # Should still be a friendly greeting
    assert "hello" in low or "help" in low

def test_guard_message_is_gated_by_greeting():
    # Greeting should be recognized
    assert main.is_greeting("Hello")
    assert main.is_greeting("good evening")
    assert not main.is_greeting("thanks")
    # Memory query should be recognized
    assert main.asks_for_memory("what do you remember?")
    assert not main.asks_for_memory("tell me a joke")
