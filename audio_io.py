import sounddevice as sd
import numpy as np

# Avoiding virtual devices for now
BLACKLIST = ["steam", "oculus", "droidcam", "virtual", "wave", "mapper"]

def _is_ok(name: str) -> bool:
    n = (name or "").lower()
    return not any(b in n for b in BLACKLIST)

def _hostapi_index(name_substring="wasapi"):
    for i, ha in enumerate(sd.query_hostapis()):
        if name_substring.lower() in ha["name"].lower():
            return i
    return None

def _default_for_hostapi(ha_index, kind="input"):
    """Return default input/output device index for a host API if usable."""
    if ha_index is None:
        return None
    ha = sd.query_hostapis()[ha_index]
    idx = ha["default_input_device"] if kind == "input" else ha["default_output_device"]
    if idx is None or idx < 0:
        return None
    dev = sd.query_devices(idx)
    if ((kind == "input" and dev["max_input_channels"] > 0) or
        (kind == "output" and dev["max_output_channels"] > 0)) and _is_ok(dev["name"]):
        return idx
    return None

def _first_usable(kind="input", prefer_hostapi="wasapi"):
    """Find the first non-blacklisted device"""
    ha_idx = _hostapi_index(prefer_hostapi)
    def usable(d, k):
        if not _is_ok(d["name"]):
            return False
        if k == "input":
            return d["max_input_channels"] > 0
        return d["max_output_channels"] > 0

    # 1) Prefer the chosen hostapi
    if ha_idx is not None:
        for i, d in enumerate(sd.query_devices()):
            if d["hostapi"] == ha_idx and usable(d, kind):
                return i
    # 2) Otherwise any hostapi
    for i, d in enumerate(sd.query_devices()):
        if usable(d, kind):
            return i
    return None

def pick_devices():
    """
    Returns: (in_idx, out_idx, in_rate, out_rate)
    Strategy: WASAPI defaults -> WASAPI first usable -> any usable.
    """
    ha = _hostapi_index("wasapi")
    in_idx  = _default_for_hostapi(ha, "input")  or _first_usable("input",  "wasapi")
    out_idx = _default_for_hostapi(ha, "output") or _first_usable("output", "wasapi")

    if in_idx is None or out_idx is None:
        # Fallback to anything
        in_idx  = in_idx  or _first_usable("input",  prefer_hostapi="")
        out_idx = out_idx or _first_usable("output", prefer_hostapi="")

    if in_idx is None or out_idx is None:
        raise RuntimeError("No suitable input/output devices found. Check Windows Sound settings.")

    in_rate  = int(sd.query_devices(in_idx)["default_samplerate"])
    out_rate = int(sd.query_devices(out_idx)["default_samplerate"])
    return in_idx, out_idx, in_rate, out_rate

def resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Very small, dependency-free resampler (mono int16)."""
    if src_rate == dst_rate:
        return x
    n_src = x.shape[0]
    n_dst = int(round(n_src * (dst_rate / src_rate)))
    # convert to float 
    x_f = x.astype(np.float32)
    src_idx = np.linspace(0, n_src - 1, num=n_src, endpoint=True)
    dst_idx = np.linspace(0, n_src - 1, num=n_dst, endpoint=True)
    y = np.interp(dst_idx, src_idx, x_f[:, 0])  # mono
    return y.astype(np.int16).reshape(-1, 1)

def record_seconds(seconds=3):
    """Auto-pick devices, record, and return (audio_int16, in_rate, out_rate, in_idx, out_idx)."""
    in_idx, out_idx, in_rate, out_rate = pick_devices()
    sd.default.device = (in_idx, out_idx)
    sd.default.channels = 1  # mono voice
    print(f"Using mic #{in_idx} '{sd.query_devices(in_idx)['name']}' @ {in_rate} Hz")
    print(f"Using spk #{out_idx} '{sd.query_devices(out_idx)['name']}' @ {out_rate} Hz")
    print(f"Recording {seconds}s...")
    audio = sd.rec(int(seconds * in_rate), samplerate=in_rate, dtype="int16")
    sd.wait()
    return audio, in_rate, out_rate, in_idx, out_idx

def play(audio: np.ndarray, in_rate: int, out_rate: int):
    """Play audio, resampling if needed to the output's native rate."""
    to_play = audio if in_rate == out_rate else resample_linear(audio, in_rate, out_rate)
    rate = out_rate
    print("Playing back...")
    sd.play(to_play, samplerate=rate)
    sd.wait()

if __name__ == "__main__":
    audio, in_rate, out_rate, in_idx, out_idx = record_seconds(3)
    play(audio, in_rate, out_rate)
    print("All good :D")
