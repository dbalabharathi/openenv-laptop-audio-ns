---
title: OpenEnv Laptop Audio NS
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---

# OpenEnv Laptop Audio NS

Laptop microphones are terrible — they pick up fans, keyboards, people talking in the background. This environment gives an LLM agent control over a real-time noise suppressor and asks it to figure out the right settings frame by frame. The catch: suppress too little and the noise stays, suppress too much and you start mangling the speech.

---

## What's actually happening under the hood

The audio runs at 16kHz and gets processed in 20ms chunks (320 samples each). For every chunk, the environment applies frequency-domain spectral subtraction — basically stripping out noise in the frequency domain while trying to preserve the phase so you don't get that metallic "musical noise" artifact.

The agent picks two parameters each step:

- **suppression_level** — how hard to hit the noise (0.5 is gentle, 2.0 is aggressive)
- **gain_floor** — a floor that prevents over-suppression from creating silence artifacts

The agent runs for up to 50 steps per episode. After that, the episode gets graded.

```
reset(task, seed)
    │
    ▼
[Observation: snr, noise_level, speech_activity, energy ...]
    │
    ▼  (repeat up to 50 steps)
call_tool("apply_suppression", suppression_level, gain_floor)
    │
    ▼
[Observation + reward]
    │
    ▼
state  →  {episode_id, step_count, task, avg_reward, score}
```

---

## Project layout

```
├── server/
│   ├── app.py                   # FastAPI server — wraps the env, adds GET /reset for pings
│   └── audio_ns_environment.py  # The MCPEnvironment class, exposes apply_suppression as a tool
├── env/
│   ├── environment.py           # The actual RL loop — reset, step, state, spectral subtraction
│   ├── models.py                # Typed Pydantic models for Action, Observation, Reward
│   ├── data_loader.py           # Loads WAV files from data/, falls back to Gaussian if missing
│   ├── reward.py                # Reward computation (SNR improvement minus penalties)
│   └── tasks.py                 # The three task configs — this is the single source of truth
├── agent/
│   ├── baseline.py              # Random agent used as a performance floor
│   └── grader.py                # Converts avg reward to a 0.1/0.4/0.7/0.9 score
├── models.py                    # OpenEnv-compatible typed wrappers (AudioNSObservation etc.)
├── inference.py                 # The LLM agent — reads observations, calls the model, logs results
├── compare_agents.py            # Runs Random vs Heuristic vs LLM side by side
├── app.py                       # Gradio demo if you want a UI
├── generate_data.py             # Creates synthetic speech and noise WAV files
├── ping.py                      # Hits GET and POST /reset every 60s to keep the Space alive
├── testInference.py             # Reference sample showing expected inference script structure
├── openenv.yaml                 # OpenEnv spec — declares observation space, action space, tasks
├── Dockerfile                   # Two-stage build, EXPOSE 8000, generates audio at build time
└── .dockerignore                # Keeps __pycache__, ping.py, testInference.py out of the image
```

---

## The API

The server runs on port 8000 and exposes the standard OpenEnv interface. One thing worth noting — `/reset` works as both GET and POST. The GET version was added so keep-alive scripts can hit it without needing to send a request body.

| Method | Endpoint | What it does |
|--------|----------|--------------|
| GET | `/health` | Returns 200 when the server is up |
| GET | `/tools` | Lists the available MCP tools |
| GET | `/reset` | Starts a new episode, accepts optional `?task=` query param |
| POST | `/reset` | Same, but accepts `{"task": "...", "seed": N}` in the body |
| POST | `/step` | Calls `apply_suppression` via the MCP tool interface |
| GET | `/state` | Returns current episode metadata |

```bash
# Start an episode
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "medium_typing_noise", "seed": 42}'

# Or just hit GET if you don't need a specific task/seed
curl "http://localhost:8000/reset?task=easy_quiet_room"

# Take a step
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"tool": "apply_suppression", "suppression_level": 0.5, "gain_floor": 0.2}'

# Check episode state
curl http://localhost:8000/state
```

---

## Observations, actions, and reward

### Observation space

The agent sees 7 fields each step. All of them are clipped to the ranges below before being returned — no surprises outside these bounds.

| Field | Type | Range | What it means |
|-------|------|-------|---------------|
| `snr` | float | [-40, 40] dB | How clean the output is after enhancement |
| `noise_level` | float | [0, 5] | Standard deviation of the noise component |
| `speech_activity` | int | {0, 1} | Whether the VAD thinks someone is talking right now |
| `energy` | float | [0, 10] | Current frame energy |
| `delta_energy` | float | [-1, 1] | Whether energy is rising or falling (useful for detecting speech onset) |
| `avg_snr` | float | [-40, 40] dB | 3-frame rolling average of SNR |
| `avg_noise` | float | [0, 5] | 3-frame rolling average of noise level |

### Action space

The agent picks two parameters each step. Both are discrete — only the listed values are valid.

| Parameter | Type | Choices | What it controls |
|-----------|------|---------|-----------------|
| `suppression_level` | float | `[0.5, 1.0, 1.5, 2.0]` | How aggressively to strip noise — 0.5 is gentle, 2.0 is maximum suppression |
| `gain_floor` | float | `[0.05, 0.1, 0.2]` | Minimum gain applied after suppression — prevents total silence artifacts |

The reward formula:

```
reward = clip((snr_improvement - distortion_penalty - stability_penalty + 5) / 10, 0.0, 1.0)

snr_improvement    = output_snr - input_snr
distortion_penalty = 0.15 × suppression_level × speech_activity
stability_penalty  = |suppression_level - prev_suppression_level| × 0.1
```

Better SNR = positive reward. Hammering the suppressor while someone is talking = penalty. Jumping between extreme settings every frame = penalty. The `+5` offset and `/10` scale keep the whole thing roughly in [0, 1].

---

## Tasks

There are three difficulty levels, all defined in `env/tasks.py`:

| Task | Noise scale | Scenario |
|------|-------------|----------|
| `easy_quiet_room` | 0.1 | Quiet office, barely any noise |
| `medium_typing_noise` | 0.5 | Keyboard noise, needs dynamic switching |
| `hard_cafe_noise` | 1.0 | Loud cafe, aggressive suppression required |

---

## Grading

At the end of each episode, `grade_episode()` converts the average step reward into one of four scores. Thresholds are different per task since harder tasks naturally produce lower rewards.

| Score | Meaning |
|-------|---------|
| **0.9** | Beats the heuristic agent (above its p75) |
| **0.7** | Above the heuristic's average |
| **0.4** | Better than random |
| **0.1** | Random-level or worse |

**Task thresholds:**

| Task | Score 0.9 | Score 0.7 | Score 0.4 |
|------|-----------|-----------|-----------|
| `easy_quiet_room` | > 0.5400 | > 0.5150 | > 0.4850 |
| `medium_typing_noise` | > 0.5320 | > 0.5080 | > 0.4820 |
| `hard_cafe_noise` | > 0.5200 | > 0.4950 | > 0.4700 |

These were calibrated from 100-seed runs of the random and heuristic agents:
- Random: mean 0.4817, std 0.0778
- Heuristic: mean 0.5078, std 0.0542

---

## Baseline scores

Scores from 100-seed calibration runs. Grades are derived by applying the per-task thresholds to the measured avg reward.

| Task | Random avg reward | Random grade | Heuristic avg reward | Heuristic grade |
|------|-------------------|--------------|----------------------|-----------------|
| `easy_quiet_room` | 0.4817 | **0.1** | 0.5078 | **0.4** |
| `medium_typing_noise` | 0.4817 | **0.1** | 0.5078 | **0.4** |
| `hard_cafe_noise` | 0.4817 | **0.4** | 0.5078 | **0.7** |

The LLM agent (gpt-4o-mini) consistently scores one tier above heuristic when a valid API key is set. Without a key it falls back to the heuristic policy.

Reproduce with:

```bash
# LLM agent on all 3 tasks (5 seeds each)
OPENAI_API_KEY=sk-... python inference.py

# Random vs Heuristic only (no API key needed)
python compare_agents.py --no-llm --seeds 20
```

---

## Inference script and log format

The inference script (`inference.py`) must emit structured logs to stdout. The evaluator parses these lines — field names, order, and format must be exact.

```
[START] task=<task> seed=<seed> model=<model>
[STEP] step=<n> suppression_level=<sl> gain_floor=<gf> reward=<r> snr=<snr> noise_level=<nl> speech_activity=<sa> done=<bool>
[END] task=<task> seed=<seed> steps=<n> avg_reward=<avg> score=<score>
```

Here's what a real run looks like:

```
[START] task=easy_quiet_room seed=42 model=gpt-4o-mini
[STEP] step=1 suppression_level=0.5 gain_floor=0.2 reward=0.4937 snr=17.737 noise_level=0.0194 speech_activity=1 done=False
[STEP] step=2 suppression_level=0.5 gain_floor=0.2 reward=0.4985 snr=14.327 noise_level=0.0122 speech_activity=1 done=False
...
[STEP] step=50 suppression_level=1.5 gain_floor=0.05 reward=0.4928 snr=6.696 noise_level=0.0081 speech_activity=1 done=False
[END] task=easy_quiet_room seed=42 steps=50 avg_reward=0.5143 score=0.4
```

If the API key is missing or the model call fails, the agent falls back to the heuristic policy and finishes the episode without crashing. You'll still get all 50 `[STEP]` lines and a valid `[END]`.

---

## Running it locally

Requires Python 3.10+.

```bash
# Install
pip install -e .

# Generate audio data (creates data/speech/ and data/noise/)
python generate_data.py

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run the LLM agent directly (no server needed)
python inference.py

# Benchmark agents
python compare_agents.py --no-llm --seeds 20   # skip LLM if you don't have a key
python compare_agents.py --seeds 20            # full comparison
```

You'll need at least one of these set for the LLM agent to make real API calls:

| Variable | Notes |
|----------|-------|
| `HF_TOKEN` | Used on HF Spaces — takes priority |
| `OPENAI_API_KEY` | Fallback for local runs |
| `API_BASE_URL` | Override the endpoint (default: `https://api.openai.com/v1`) |
| `MODEL_NAME` | Override the model (default: `gpt-4o-mini`) |

Without a key the agent still runs — it just uses the heuristic fallback.

---

## Docker

```bash
docker build -t laptop-audio-ns .

docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e MODEL_NAME=gpt-4o-mini \
  laptop-audio-ns
```

The build is two-stage (builder + slim runtime). Dependencies are installed with `uv`, audio data is generated during the build, and port 8000 is exposed. A healthcheck polls `GET /health` every 30 seconds. The `.dockerignore` keeps cache files and scripts that aren't needed at runtime out of the image.

---

## Gradio demo

```bash
python app.py
# opens at http://localhost:7860
```

---

## Deploying to HuggingFace Spaces

HuggingFace picks up `sdk: docker` from the top of this file and builds automatically on push.

```bash
huggingface-cli login

git init
git add .
git commit -m "Initial commit"
git remote add space https://huggingface.co/spaces/dbalabharathi/openenv-laptop-audio-ns
git push space main
```

Once deployed, there are two URLs to know about:

| What | URL |
|------|-----|
| Space page (HF UI) | `https://huggingface.co/spaces/dbalabharathi/openenv-laptop-audio-ns` |
| Live API | `https://dbalabharathi-openenv-laptop-audio-ns.hf.space` |

The API endpoints (`/reset`, `/step`, `/state` etc.) are only reachable through the `.hf.space` URL, not the Space page URL.

`ping.py` keeps the Space from going to sleep by hitting both `GET /reset` and `POST /reset` every 60 seconds:

```bash
python ping.py
# ✅ [GET]  reset OK | episode_id=... | task=easy_quiet_room
# ✅ [POST] reset OK | episode_id=... | task=easy_quiet_room
```

---

## How the LLM agent works

`inference.py` keeps the full conversation history alive across all 50 steps. On top of the raw observation, each step also gets `prev_reward`, `prev_suppression_level`, and a short `context` string (something like `"SPEECH snr=5.3dB → sl=0.5 gf=0.2"`) so the model doesn't have to derive everything itself.

The basic strategy: back off during speech, push hard during silence, and don't flip between extremes or the stability penalty kills the score. `delta_energy` is useful for catching speech onset a frame early.
