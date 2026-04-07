# Thresholds calibrated from 100-seed runs of Random and Heuristic baselines.
# Grade tiers: 0.9 = clearly above heuristic p75
#              0.7 = above heuristic mean
#              0.4 = above random mean
#              0.1 = random-level or worse

TASK_THRESHOLDS = {
    "easy_quiet_room":     {"t10": 0.5400, "t07": 0.5150, "t04": 0.4850},
    "medium_typing_noise": {"t10": 0.5320, "t07": 0.5080, "t04": 0.4820},
    "hard_cafe_noise":     {"t10": 0.5200, "t07": 0.4950, "t04": 0.4700},
}


def grade_episode(rewards: list, task: str = "medium_typing_noise") -> float:
    if not rewards:
        return 0.1
    avg = sum(rewards) / len(rewards)
    t = TASK_THRESHOLDS.get(task, TASK_THRESHOLDS["medium_typing_noise"])
    if avg > t["t10"]:   return 0.9
    elif avg > t["t07"]: return 0.7
    elif avg > t["t04"]: return 0.4
    else:                return 0.1
