import os
import json
import random
import numpy as np
from openai import OpenAI
from env.environment import LaptopAudioEnv, MAX_STEPS
from env.models import Action
from agent.grader import grade_episode

MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ.get("HF_TOKEN") or os.environ.get("API_KEY"),
            base_url=os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
        )
    return _client


SYSTEM_PROMPT = """You control a noise suppressor for a laptop microphone. Each frame, pick:
  suppression_level: one of [0.5, 1.0, 1.5, 2.0]
  gain_floor:        one of [0.05, 0.1, 0.2]

Reward per frame:
  reward = (snr_improvement - distortion_penalty - stability_penalty + 5) / 10
  snr_improvement    = output_snr - input_snr      (more suppression → more noise removed)
  distortion_penalty = 0.15 * suppression_level * speech_activity
  stability_penalty  = |suppression_level - prev_suppression_level| * 0.1

Observation fields:
  snr              output SNR in dB after your last action (higher = cleaner audio)
  noise_level      std of the noise component (higher = noisier)
  speech_activity  1 if speech detected, 0 if silence
  energy           current frame energy
  delta_energy     energy change vs last frame (positive = volume rising)
  avg_snr          3-frame rolling SNR
  avg_noise        3-frame rolling noise level

Strategy:
  During SPEECH (speech_activity=1):
    - Default: sl=0.5, gf=0.2 to keep distortion_penalty low
    - If snr < 0 and prev_reward < 0.50, sl=1.0 may be worth the extra distortion

  During SILENCE (speech_activity=0):
    - snr < 0   → sl=2.0, gf=0.05  (heavy noise, maximum suppression)
    - snr 0–8   → sl=1.5, gf=0.05  (moderate noise)
    - snr > 8   → sl=1.0, gf=0.05  (already clean, don't over-process)

  Transitions:
    - delta_energy > 0.02   → speech starting soon, drop to sl=0.5 now
    - delta_energy < -0.02  → speech ending, start raising suppression

Watch prev_reward — if it's below 0.50 your last action underperformed, reconsider.

Reply with ONLY valid JSON: {"suppression_level": <value>, "gain_floor": <value>}"""

SUPPRESSION_CHOICES = [0.5, 1.0, 1.5, 2.0]
GAIN_FLOOR_CHOICES  = [0.05, 0.1, 0.2]
HISTORY_WINDOW      = 6  # last 3 turns — enough context without bloating the prompt


def _snap(value, choices):
    return min(choices, key=lambda x: abs(x - value))


def _fallback(obs) -> Action:
    if obs.speech_activity:
        return Action(suppression_level=0.5, gain_floor=0.2)
    sl = 2.0 if obs.avg_noise > 0.5 else 1.5
    return Action(suppression_level=sl, gain_floor=0.05)


def _context(obs, prev_reward, prev_sl, step, avg_reward) -> str:
    hints = []

    if obs.speech_activity:
        if obs.snr < 0:
            hints.append(f"SPEECH snr={obs.snr:.1f}dB (negative — try sl=1.0 if prev_reward low)")
        else:
            hints.append(f"SPEECH snr={obs.snr:.1f}dB → sl=0.5 gf=0.2")
    else:
        if obs.snr < 0:
            hints.append(f"SILENCE snr={obs.snr:.1f}dB (very noisy) → sl=2.0 gf=0.05")
        elif obs.snr < 8:
            hints.append(f"SILENCE snr={obs.snr:.1f}dB → sl=1.5 gf=0.05")
        else:
            hints.append(f"SILENCE snr={obs.snr:.1f}dB (clean) → sl=1.0 sufficient")

    if prev_sl is not None:
        costs = {sl: round(abs(sl - prev_sl) * 0.1, 3) for sl in SUPPRESSION_CHOICES}
        hints.append("stability costs: " + ", ".join(f"sl={sl}→{c}" for sl, c in costs.items()))

    if obs.delta_energy > 0.02:
        hints.append(f"energy rising ({obs.delta_energy:.3f}) — speech onset, prepare sl=0.5")
    elif obs.delta_energy < -0.02:
        hints.append(f"energy falling ({obs.delta_energy:.3f}) — can increase suppression")

    if prev_reward is not None:
        status = "good" if prev_reward >= 0.53 else ("ok" if prev_reward >= 0.50 else "suboptimal — reconsider")
        hints.append(f"prev_reward={prev_reward:.3f} ({status})")

    if step is not None:
        hints.append(f"step {step}/{MAX_STEPS}")
    if avg_reward is not None:
        hints.append(f"episode avg={avg_reward:.3f} ({'on track' if avg_reward >= 0.508 else 'below target'})")

    return " | ".join(hints)


def get_action(obs, history, prev_reward=None, prev_action=None, step=None, avg_reward=None) -> Action:
    prev_sl = prev_action.suppression_level if prev_action else None

    user_msg = json.dumps({
        "snr":                    round(obs.snr, 3),
        "noise_level":            round(obs.noise_level, 3),
        "speech_activity":        obs.speech_activity,
        "energy":                 round(obs.energy, 6),
        "delta_energy":           round(obs.delta_energy, 6),
        "avg_snr":                round(obs.avg_snr, 3),
        "avg_noise":              round(obs.avg_noise, 3),
        "prev_reward":            round(prev_reward, 4) if prev_reward is not None else None,
        "prev_suppression_level": prev_sl,
        "context":                _context(obs, prev_reward, prev_sl, step, avg_reward),
    })

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history[-HISTORY_WINDOW:]
        + [{"role": "user", "content": user_msg}]
    )

    try:
        resp = _get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_completion_tokens=50,
        )
        text   = resp.choices[0].message.content.strip()
        data   = json.loads(text)
        sl     = _snap(float(data["suppression_level"]), SUPPRESSION_CHOICES)
        gf     = _snap(float(data["gain_floor"]), GAIN_FLOOR_CHOICES)
        action = Action(suppression_level=sl, gain_floor=gf)
        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant", "content": text})
        return action

    except Exception:
        action = _fallback(obs)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": json.dumps({
            "suppression_level": action.suppression_level,
            "gain_floor":        action.gain_floor,
        })})
        return action


def run(task: str, seed: int = None) -> float:
    if seed is None:
        seed = random.randint(0, 99999)
    random.seed(seed)
    np.random.seed(seed)

    env     = LaptopAudioEnv(task)
    obs     = env.reset(seed=seed)
    rewards, history = [], []
    prev_reward = prev_action = None

    print(f"[START] task={task} seed={seed} model={MODEL}")

    for step in range(1, MAX_STEPS + 1):
        avg_so_far = sum(rewards) / len(rewards) if rewards else None
        action     = get_action(obs, history, prev_reward, prev_action, step=step, avg_reward=avg_so_far)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward.value)
        prev_reward = reward.value
        prev_action = action
        env.state()

        print(
            f"[STEP] step={step} "
            f"suppression_level={action.suppression_level} "
            f"gain_floor={action.gain_floor} "
            f"reward={reward.value:.4f} "
            f"snr={obs.snr:.3f} "
            f"noise_level={obs.noise_level:.4f} "
            f"speech_activity={obs.speech_activity} "
            f"done={done}"
        )

        if done:
            break

    score      = grade_episode(rewards, task)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"[END] task={task} seed={seed} steps={len(rewards)} avg_reward={avg_reward:.4f} score={score}")

    env.close()
    return score


def run_multi(task: str, n: int = 5, seeds: list = None) -> dict:
    if seeds is None:
        seeds = [random.randint(0, 99999) for _ in range(n)]
    scores = []
    for s in seeds:
        score = run(task, seed=s)
        scores.append((s, score))
        print(f"  seed={s:<8} score={score}")
    avg = sum(s for _, s in scores) / len(scores)
    print(f"  avg={avg:.3f} over {len(seeds)} runs")
    return {"scores": scores, "avg": avg}


if __name__ == "__main__":
    for task in ["easy_quiet_room", "medium_typing_noise", "hard_cafe_noise"]:
        print(f"\n{task}:")
        run_multi(task, n=5)
