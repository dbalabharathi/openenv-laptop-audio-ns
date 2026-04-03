import os
import numpy as np

SAMPLE_RATE = 16000
_speech_cache = []
_noise_cache  = []


def _load_wavs(directory):
    clips = []
    if not os.path.isdir(directory):
        return clips
    try:
        import soundfile as sf
    except ImportError:
        return clips
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith('.wav'):
            continue
        try:
            data, sr = sf.read(os.path.join(directory, fname), dtype='float32')
            if data.ndim > 1:
                data = data[:, 0]
            if sr != SAMPLE_RATE:
                # naive integer resampling — good enough for synthetic data
                idx  = np.arange(0, len(data), sr / SAMPLE_RATE).astype(int)
                data = data[idx[idx < len(data)]]
            clips.append(data.astype(np.float64))
        except Exception:
            pass
    return clips


def _pick_clip(clips, seed, length):
    if not clips:
        return None
    rng   = np.random.RandomState(seed)
    audio = clips[rng.randint(len(clips))]
    if len(audio) < length:
        audio = np.tile(audio, length // len(audio) + 1)
    start = rng.randint(0, len(audio) - length + 1)
    return audio[start:start + length].copy()


def load_audio_pair(noise_scale: float, seed: int, target_len: int = 80_000):
    global _speech_cache, _noise_cache
    if not _speech_cache:
        _speech_cache = _load_wavs("data/speech")
    if not _noise_cache:
        _noise_cache  = _load_wavs("data/noise")

    rng = np.random.RandomState(seed)

    clean = _pick_clip(_speech_cache, seed,     target_len)
    noise = _pick_clip(_noise_cache,  seed + 1, target_len)

    # fall back to synthetic Gaussian if WAV files are missing
    if clean is None: clean = rng.randn(target_len)
    if noise is None: noise = rng.randn(target_len)

    clean = clean / (np.max(np.abs(clean)) + 1e-8)
    noise = noise / (np.max(np.abs(noise)) + 1e-8)

    return clean, noise_scale * noise
