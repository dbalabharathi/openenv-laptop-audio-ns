"""
Generate synthetic speech-like and noise WAV files for the audio NS environment.

  - Speech: harmonic series with syllabic envelope and silence gaps
  - Noise: white, pink (1/f), and keyboard-impulse types

Usage:
    python generate_data.py
"""

import os
import numpy as np

SAMPLE_RATE = 16000


def generate_speech(duration=2.0, seed=0):
    rng = np.random.RandomState(seed)
    n   = int(duration * SAMPLE_RATE)
    t   = np.arange(n) / SAMPLE_RATE

    f0     = rng.uniform(120, 200)
    voiced = sum(
        (1.0 / h) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))
        for h in range(1, 9)
    )

    # syllabic amplitude modulation (~3–6 Hz)
    rate     = rng.uniform(3.0, 6.0)
    envelope = np.clip((np.sin(2 * np.pi * rate * t) + 1.2) / 2.2, 0.05, 1.0)
    speech   = voiced * envelope

    # random silence gaps (~10% of frames)
    speech[rng.rand(n) < 0.10] = 0.0
    speech /= np.max(np.abs(speech)) + 1e-8
    return speech.astype(np.float32)


def generate_noise(noise_type="pink", duration=2.0, seed=0):
    rng = np.random.RandomState(seed)
    n   = int(duration * SAMPLE_RATE)

    if noise_type == "white":
        noise = rng.randn(n).astype(np.float32)

    elif noise_type == "pink":
        f = np.fft.rfftfreq(n)
        f[0] = 1.0  # avoid divide-by-zero at DC
        spectrum = (rng.randn(len(f)) + 1j * rng.randn(len(f))) / np.sqrt(f)
        noise = np.fft.irfft(spectrum, n=n).astype(np.float32)

    elif noise_type == "keyboard":
        noise = (0.08 * rng.randn(n)).astype(np.float32)
        for _ in range(int(rng.uniform(2, 6) * duration)):
            t0  = rng.randint(0, n)
            end = min(t0 + int(rng.uniform(0.01, 0.03) * SAMPLE_RATE), n)
            noise[t0:end] += (rng.randn(end - t0) * rng.uniform(0.5, 1.0)).astype(np.float32)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}")

    noise /= np.max(np.abs(noise)) + 1e-8
    return noise


if __name__ == "__main__":
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile not installed. Run: pip install soundfile")
        raise SystemExit(1)

    os.makedirs("data/speech", exist_ok=True)
    os.makedirs("data/noise",  exist_ok=True)

    print("Generating speech samples...")
    for i in range(10):
        sf.write(f"data/speech/speech_{i:03d}.wav", generate_speech(duration=5.0, seed=i), SAMPLE_RATE)
    print("  10 files → data/speech/")

    print("Generating noise samples...")
    idx = 0
    for ntype in ["white", "pink", "keyboard"]:
        for j in range(4):
            sf.write(f"data/noise/noise_{idx:03d}_{ntype}.wav", generate_noise(ntype, duration=5.0, seed=idx), SAMPLE_RATE)
            idx += 1
    print(f"  {idx} files → data/noise/")
    print("Done.")
