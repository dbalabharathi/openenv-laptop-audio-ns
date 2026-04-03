from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State
from fastmcp import FastMCP

from env.environment import LaptopAudioEnv, MAX_STEPS, SUPPRESSION_CHOICES, GAIN_FLOOR_CHOICES
from env.models import Action as EnvAction
from agent.grader import grade_episode
from models import AudioNSObservation


class LaptopAudioNSEnvironment(MCPEnvironment):
    """
    OpenEnv server for the adaptive noise suppression task.

    The agent interacts via one MCP tool: apply_suppression(suppression_level, gain_floor).
    Each call advances the environment by one 20ms audio frame and returns
    the resulting observation + reward.
    """

    def __init__(self):
        # shared state dict — closures inside __init__ reference this before super().__init__()
        self._sc = {
            "env":         None,
            "task":        "medium_typing_noise",
            "step_count":  0,
            "episode_id":  str(uuid4()),
            "done":        False,
            "rewards":     [],
            "last_reward": 0.0,
        }
        sc  = self._sc
        mcp = FastMCP("laptop_audio_ns")

        @mcp.tool
        def apply_suppression(suppression_level: float, gain_floor: float) -> dict:
            """Apply noise suppression to the current 20ms audio frame.

            Args:
                suppression_level: aggressiveness — one of [0.5, 1.0, 1.5, 2.0]
                gain_floor:        minimum gain floor — one of [0.05, 0.1, 0.2]

            Returns:
                snr, noise_level, speech_activity, energy, delta_energy,
                avg_snr, avg_noise, reward, done, step
            """
            if sc["env"] is None:
                return {"error": "Call reset() before stepping."}
            if sc["done"]:
                return {"error": "Episode finished. Call reset() to start a new one."}

            sl = min(SUPPRESSION_CHOICES, key=lambda x: abs(x - suppression_level))
            gf = min(GAIN_FLOOR_CHOICES,  key=lambda x: abs(x - gain_floor))

            obs, reward, done, _ = sc["env"].step(EnvAction(suppression_level=sl, gain_floor=gf))

            sc["step_count"] += 1
            sc["done"]        = done
            sc["last_reward"] = reward.value
            sc["rewards"].append(reward.value)

            return {
                "snr":                round(obs.snr, 3),
                "noise_level":        round(obs.noise_level, 4),
                "speech_activity":    obs.speech_activity,
                "energy":             round(obs.energy, 6),
                "delta_energy":       round(obs.delta_energy, 6),
                "avg_snr":            round(obs.avg_snr, 3),
                "avg_noise":          round(obs.avg_noise, 4),
                "reward":             round(reward.value, 4),
                "snr_improvement":    reward.snr_improvement,
                "distortion_penalty": reward.distortion_penalty,
                "stability_penalty":  reward.stability_penalty,
                "done":               done,
                "step":               sc["step_count"],
            }

        super().__init__(mcp)

    def reset(self, seed=None, episode_id=None, task=None, **kwargs) -> Observation:
        task_name = task or self._sc["task"]
        core_env  = LaptopAudioEnv(task_name)
        obs       = core_env.reset(seed=seed)

        self._sc.update({
            "env":         core_env,
            "task":        task_name,
            "step_count":  0,
            "episode_id":  episode_id or str(uuid4()),
            "done":        False,
            "rewards":     [],
            "last_reward": 0.0,
        })

        return AudioNSObservation(
            snr=obs.snr, noise_level=obs.noise_level, speech_activity=obs.speech_activity,
            energy=obs.energy, delta_energy=obs.delta_energy, avg_snr=obs.avg_snr, avg_noise=obs.avg_noise,
            reward=0.0, done=False, step=0,
            episode_id=self._sc["episode_id"], task=task_name,
        )

    def _step_impl(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        # all actions must come through the MCP tool, not here
        return AudioNSObservation(
            snr=0.0, noise_level=0.0, speech_activity=0,
            energy=0.0, delta_energy=0.0, avg_snr=0.0, avg_noise=0.0,
            reward=0.0, done=True, step=self._sc["step_count"],
            episode_id=self._sc["episode_id"], task=self._sc["task"],
            error="Use call_tool('apply_suppression', ...) instead.",
        )

    @property
    def state(self) -> State:
        rewards = self._sc["rewards"]
        score   = grade_episode(rewards, self._sc["task"]) if rewards else 0.0
        return State(
            episode_id=self._sc["episode_id"],
            step_count=self._sc["step_count"],
            extra={
                "task":        self._sc["task"],
                "done":        self._sc["done"],
                "last_reward": round(self._sc["last_reward"], 4),
                "avg_reward":  round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
                "score":       score,
                "max_steps":   MAX_STEPS,
            },
        )
