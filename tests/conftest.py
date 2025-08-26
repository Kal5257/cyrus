import os
import sys
from pathlib import Path
import types
import pytest

# Ensure repo root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture(autouse=True)
def temp_data_dir(tmp_path, monkeypatch):
    """Redirect memory persistence to a temp folder for every test."""
    monkeypatch.setenv("CYRUS_DATA_DIR", str(tmp_path))

@pytest.fixture(scope="session", autouse=True)
def stub_heavy_modules_before_main_import():
    """
    Stub out heavy/CI-unavailable modules so importing main.py is cheap and offline:
    - sounddevice, soundfile, piper, faster_whisper
    """
    # ---- sounddevice stub ----
    sd = types.SimpleNamespace()

    # sd.query_devices(None,'output') -> dict with default samplerate
    def _query_devices(index=None, kind=None):
        if kind == 'output':
            return {'default_samplerate': 48000}
        # return a minimal list/dict when called differently (we don't use others in main)
        return {'default_samplerate': 48000}
    sd.query_devices = _query_devices
    sd.play = lambda *a, **k: None
    sd.wait = lambda *a, **k: None
    sd.rec = lambda *a, **k: None
    sd.default = types.SimpleNamespace(device=None, channels=None)

    sys.modules.setdefault("sounddevice", sd)

    # ---- soundfile stub ----
    sf = types.SimpleNamespace()
    # Not used unless speak() runs fully; still provide a minimal read
    sf.read = lambda *a, **k: ([0.0], 48000)
    sys.modules.setdefault("soundfile", sf)

    # ---- piper stub ----
    p = types.SimpleNamespace()
    class FakeVoice:
        config = types.SimpleNamespace(sample_rate=22050)
        def synthesize(self, text, wf):
            # Write a tiny valid WAV frame (1 sample of silence) when used
            wf.writeframes(b"\x00\x00")
    p.PiperVoice = types.SimpleNamespace(load=lambda *args, **kwargs: FakeVoice())
    sys.modules.setdefault("piper", p)

    # ---- faster_whisper stub ----
    fw = types.SimpleNamespace()
    class FakeWhisperModel:
        def __init__(self, *a, **k): pass
        def transcribe(self, *a, **k):
            # Return segments iterator + info; main concatenates s.text
            class Seg: 
                def __init__(self, t): self.text = t
            return [Seg("hello")], None
    fw.WhisperModel = FakeWhisperModel
    sys.modules.setdefault("faster_whisper", fw)
