
import random
import numpy as np
from env.environment import LaptopAudioEnv, MAX_STEPS
from agent.grader import grade_episode


def run(task, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    env = LaptopAudioEnv(task)
    obs = env.reset(seed=seed)
    rewards = []

    for _ in range(MAX_STEPS):
        action = env.action_space.sample()          # uses ActionSpace.sample()
        obs, reward, done, _ = env.step(action)
        rewards.append(reward.value)
        if done:
            break

    env.close()
    return grade_episode(rewards, task)


if __name__ == "__main__":
    for t in ["easy_quiet_room", "medium_typing_noise", "hard_cafe_noise"]:
        print(t, run(t))
