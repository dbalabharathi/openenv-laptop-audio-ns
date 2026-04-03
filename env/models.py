import random
from dataclasses import dataclass, field
from typing import List
from pydantic import BaseModel


class Observation(BaseModel):
    snr: float
    noise_level: float
    speech_activity: int  # 1=speech, 0=silence
    energy: float
    delta_energy: float
    avg_snr: float
    avg_noise: float


class Action(BaseModel):
    suppression_level: float
    gain_floor: float


class Reward(BaseModel):
    value: float
    snr_improvement: float
    distortion_penalty: float
    stability_penalty: float


class State(BaseModel):
    step: int
    prev_snr: float


@dataclass
class ActionSpace:
    suppression_choices: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    gain_floor_choices: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])

    @property
    def n_actions(self) -> int:
        return len(self.suppression_choices) * len(self.gain_floor_choices)

    def sample(self) -> Action:
        return Action(
            suppression_level=random.choice(self.suppression_choices),
            gain_floor=random.choice(self.gain_floor_choices),
        )


@dataclass
class ObservationSpace:
    fields: List[str] = field(default_factory=lambda: [
        "snr", "noise_level", "speech_activity",
        "energy", "delta_energy", "avg_snr", "avg_noise",
    ])
    low: List[float] = field(default_factory=lambda: [-40.0, 0.0, 0, 0.0, -1.0, -40.0, 0.0])
    high: List[float] = field(default_factory=lambda: [40.0, 5.0, 1, 10.0, 1.0, 40.0, 5.0])

    @property
    def n_fields(self) -> int:
        return len(self.fields)
