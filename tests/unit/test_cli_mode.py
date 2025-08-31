# tests/unit/test_cli_mode.py
# 1) Stub heavy deps BEFORE importing main
import sys, types
from importlib import import_module

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

# piper stub (should NOT be used when --mute, but safe to have)
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

def import_main_with_argv(argv):
    """Import a fresh copy of main with custom argv."""
    sys.modules.pop("main", None)
    old_argv = sys.argv[:]
    sys.argv = argv[:]  # argparse in main reads sys.argv at import-time
    try:
        m = import_module("main")
    finally:
        sys.argv = old_argv
    return m

def test_text_input_and_mute_disables_tts_and_sets_args(capsys):
    # Simulate: python main.py --input text --mute
    m = import_main_with_argv(["main.py", "--input", "text", "--mute"])

    # Flags applied at import time
    assert m.args.input == "text"
    assert m.HAS_PIPER is False  # mute forces TTS off

    # speak() should be a safe no-op that prints the skip line
    m.speak("Hello there.")
    out = capsys.readouterr().out.lower()
    assert "(speak skipped)" in out
    assert "hello there." in out

def test_helpers_still_function_under_cli_mode():
    m = import_main_with_argv(["main.py", "--input", "text", "--mute"])
    assert m.is_greeting("Hello")
    assert not m.is_greeting("tell me a joke")
    assert m.asks_for_memory("what do you remember?")
    assert not m.asks_for_memory("open notepad")
