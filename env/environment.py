import numpy as np
from collections import deque
from .models import Observation, Action, Reward, State, ActionSpace, ObservationSpace
from .reward import compute_reward
from env.tasks import TASKS
from env.data_loader import load_audio_pair

SUPPRESSION_CHOICES = [0.5, 1.0, 1.5, 2.0]
GAIN_FLOOR_CHOICES  = [0.05, 0.1, 0.2]
MAX_STEPS           = 50
AUDIO_LEN           = 80_000  # 5s at 16kHz


def _validate_action(action: Action):
    if action.suppression_level not in SUPPRESSION_CHOICES:
        raise ValueError(f"suppression_level must be one of {SUPPRESSION_CHOICES}")
    if action.gain_floor not in GAIN_FLOOR_CHOICES:
        raise ValueError(f"gain_floor must be one of {GAIN_FLOOR_CHOICES}")


def _spectral_subtract(frame: np.ndarray, sl: float, gf: float) -> np.ndarray:
    n            = len(frame)
    spec         = np.fft.rfft(frame)
    mag, phase   = np.abs(spec), np.angle(spec)
    noise_floor  = np.percentile(mag, 20)  # 20th percentile ≈ rough noise floor
    clean_mag    = np.maximum(mag - sl * noise_floor, gf * mag)
    return np.fft.irfft(clean_mag * np.exp(1j * phase), n=n)


def _vad(frame: np.ndarray, threshold_db: float = -30.0) -> int:
    rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
    return int(20.0 * np.log10(rms) > threshold_db)


class LaptopAudioEnv:
    def __init__(self, task_name: str):
        cfg              = TASKS[task_name]
        self.noise_scale = cfg["noise"]
        self.frame_size  = 320
        self.hop         = 160

        self.ptr         = 0
        self._ep_state   = None
        self.snr_hist    = deque(maxlen=3)
        self.noise_hist  = deque(maxlen=3)
        self.prev_energy = 0.0
        self.prev_action = None
        self.clean = self.noise = self.mix = None

        self.observation_space = ObservationSpace()
        self.action_space      = ActionSpace()

    def reset(self, seed: int = None) -> Observation:
        if seed is None:
            seed = np.random.randint(0, 99999)

        self.clean, self.noise = load_audio_pair(self.noise_scale, seed, target_len=AUDIO_LEN)
        self.mix         = self.clean + self.noise
        self.ptr         = 0
        self._ep_state   = State(step=0, prev_snr=0.0)
        self.snr_hist.clear()
        self.noise_hist.clear()
        self.prev_action = None

        obs              = self._obs_at(self.ptr)
        self.prev_energy = obs.energy
        return obs

    def step(self, action: Action):
        _validate_action(action)
        self._ep_state.step += 1

        start, end = self.ptr, self.ptr + self.frame_size
        if end >= len(self.mix):
            return (
                self._obs_at(self.ptr),
                Reward(value=0.0, snr_improvement=0.0, distortion_penalty=0.0, stability_penalty=0.0),
                True,
                {},
            )

        mix_frame   = self.mix[start:end]
        clean_frame = self.clean[start:end]
        noise_frame = mix_frame - clean_frame

        enhanced     = _spectral_subtract(mix_frame, action.suppression_level, action.gain_floor)
        energy       = float(np.clip(np.mean(enhanced ** 2), 0.0, 10.0))
        delta_energy = float(np.clip(energy - self.prev_energy, -1.0, 1.0))
        self.prev_energy = energy

        noise_level = float(np.clip(np.std(noise_frame), 0.0, 5.0))
        vad         = _vad(clean_frame)

        input_snr = float(10.0 * np.log10(
            (np.mean(clean_frame ** 2) + 1e-6) / (np.mean(noise_frame ** 2) + 1e-6)
        ))
        output_snr = float(10.0 * np.log10(
            (np.mean(clean_frame ** 2) + 1e-6) / (np.mean((clean_frame - enhanced) ** 2) + 1e-6)
        ))

        self.snr_hist.append(output_snr)
        self.noise_hist.append(noise_level)

        reward               = compute_reward(input_snr, output_snr, action, self.prev_action, vad)
        self._ep_state.prev_snr = output_snr
        self.prev_action     = action
        self.ptr            += self.hop
        done                 = self.ptr + self.frame_size >= len(self.mix)

        obs = Observation(
            snr=float(np.clip(output_snr, -40.0, 40.0)),
            noise_level=noise_level,
            speech_activity=vad,
            energy=energy,
            delta_energy=delta_energy,
            avg_snr=float(np.clip(np.mean(self.snr_hist), -40.0, 40.0)),
            avg_noise=float(np.clip(np.mean(self.noise_hist), 0.0, 5.0)),
        )
        return obs, reward, done, {}

    def state(self) -> State:
        return self._ep_state

    def render(self) -> str:
        if self._ep_state is None:
            print("[LaptopAudioEnv] call reset() first")
            return ""

        snr   = self.snr_hist[-1]   if self.snr_hist   else 0.0
        noise = self.noise_hist[-1] if self.noise_hist else 0.0
        BAR   = 20

        snr_fill   = int(np.clip((snr + 10) / 40 * BAR, 0, BAR))
        noise_fill = int(np.clip(noise / 1.0 * BAR, 0, BAR))
        snr_bar    = "#" * snr_fill   + "." * (BAR - snr_fill)
        noise_bar  = "#" * noise_fill + "." * (BAR - noise_fill)

        if self.clean is not None and self.ptr > 0:
            end       = min(self.ptr, len(self.clean))
            vad_label = "[SPEECH ]" if _vad(self.clean[max(0, end - self.frame_size):end]) else "[SILENCE]"
        else:
            vad_label = "[UNKNOWN]"

        step   = self._ep_state.step
        output = "\n".join([
            f"step={step:<4} SNR  |{snr_bar}| {snr:+.1f}dB",
            f"         Noise|{noise_bar}| {noise:.3f}",
            f"         VAD  {vad_label}",
        ])
        print(output)
        return output

    def close(self):
        self.clean = self.noise = self.mix = None
        self._ep_state   = None
        self.prev_action = None
        self.prev_energy = 0.0
        self.ptr         = 0
        self.snr_hist.clear()
        self.noise_hist.clear()

    def _obs_at(self, ptr: int) -> Observation:
        end = ptr + self.frame_size
        if end > len(self.mix):
            return Observation(snr=0.0, noise_level=0.0, speech_activity=0,
                               energy=0.0, delta_energy=0.0, avg_snr=0.0, avg_noise=0.0)

        mix_frame   = self.mix[ptr:end]
        clean_frame = self.clean[ptr:end]
        noise_frame = mix_frame - clean_frame

        energy      = float(np.clip(np.mean(mix_frame ** 2), 0.0, 10.0))
        noise_level = float(np.clip(np.std(noise_frame), 0.0, 5.0))
        vad         = _vad(clean_frame)
        snr         = float(10.0 * np.log10(
            (np.mean(clean_frame ** 2) + 1e-6) / (np.mean(noise_frame ** 2) + 1e-6)
        ))

        self.snr_hist.append(snr)
        self.noise_hist.append(noise_level)

        return Observation(
            snr=float(np.clip(snr, -40.0, 40.0)),
            noise_level=noise_level,
            speech_activity=vad,
            energy=energy,
            delta_energy=0.0,
            avg_snr=float(np.clip(np.mean(self.snr_hist), -40.0, 40.0)),
            avg_noise=float(np.clip(np.mean(self.noise_hist), 0.0, 5.0)),
        )
