from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

try:
    from .audio_ns_environment import LaptopAudioNSEnvironment
except ImportError:
    from server.audio_ns_environment import LaptopAudioNSEnvironment

app = create_app(
    LaptopAudioNSEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="laptop_audio_ns",
)


@app.get("/reset")
def reset_get(task: str = "easy_quiet_room"):
    """GET /reset — lightweight alias for keep-alive pings and quick environment checks."""
    env = LaptopAudioNSEnvironment()
    obs = env.reset(task=task)
    return obs.model_dump()


import gradio as gr
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import build_ui

app = gr.mount_gradio_app(app, build_ui(), path="/")


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
