import gradio as gr
from inference import run, run_multi


def evaluate(task, mode, n_runs):
    if mode == "Single run (random seed)":
        score = run(task)
        return f"Score: {score}"
    else:
        results = run_multi(task, n=int(n_runs))
        lines = [f"seed={s:<8}  score={sc}" for s, sc in results["scores"]]
        lines.append(f"\nAvg score: {results['avg']:.3f} over {int(n_runs)} runs")
        return "\n".join(lines)


def build_ui():
    return gr.Interface(
        fn=evaluate,
        inputs=[
            gr.Dropdown(
                ["easy_quiet_room", "medium_typing_noise", "hard_cafe_noise"],
                label="Task",
            ),
            gr.Radio(
                ["Single run (random seed)", "Multi-seed (variance test)"],
                value="Single run (random seed)",
                label="Mode",
            ),
            gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of runs (multi-seed mode)"),
        ],
        outputs=gr.Textbox(label="Result", lines=12),
        title="Laptop Mic Noise Suppression — LLM Agent",
        description="LLM-driven adaptive noise suppression. Single run uses a random seed; multi-seed shows score variance.",
    )


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
