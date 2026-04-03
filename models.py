from typing import Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel


class AudioNSReward(BaseModel):
    value: float = Field(..., description="Step reward in [0, 1]")
    snr_improvement: float    = Field(default=0.0)
    distortion_penalty: float = Field(default=0.0)
    stability_penalty: float  = Field(default=0.0)


class AudioNSAction(Action):
    suppression_level: float = Field(..., description="One of [0.5, 1.0, 1.5, 2.0]")
    gain_floor: float        = Field(..., description="One of [0.05, 0.1, 0.2]")


class AudioNSObservation(Observation):
    snr: float           = Field(default=0.0)
    noise_level: float   = Field(default=0.0)
    speech_activity: int = Field(default=0)
    energy: float        = Field(default=0.0)
    delta_energy: float  = Field(default=0.0)
    avg_snr: float       = Field(default=0.0)
    avg_noise: float     = Field(default=0.0)
    reward: float        = Field(default=0.0)
    done: bool           = Field(default=False)
    step: int            = Field(default=0)
    episode_id: str      = Field(default="")
    task: str            = Field(default="")
    error: Optional[str] = Field(default=None)
