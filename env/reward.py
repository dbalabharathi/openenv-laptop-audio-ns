from .models import Reward


def compute_reward(input_snr, output_snr, action, prev_action, vad) -> Reward:
    snr_improvement    = output_snr - input_snr
    distortion_penalty = 0.15 * action.suppression_level * vad
    stability_penalty  = (
        abs(action.suppression_level - prev_action.suppression_level) * 0.1
        if prev_action is not None else 0.0
    )

    value = (snr_improvement - distortion_penalty - stability_penalty + 5) / 10
    value = max(0.0, min(1.0, value))

    return Reward(
        value=value,
        snr_improvement=round(snr_improvement, 4),
        distortion_penalty=round(distortion_penalty, 4),
        stability_penalty=round(stability_penalty, 4),
    )
