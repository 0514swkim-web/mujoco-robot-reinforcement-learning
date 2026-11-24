from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

import diff_drive_env  # 2륜 이동로봇 환경 등록

MODEL_PATH = Path(__file__).with_name("ppo_diff_drive")

env = gym.make(
    "DiffDriveGoal-v0",
    render_mode="human"
)

model = PPO.load(MODEL_PATH)

for episode in range(10):
    observation, info = env.reset()

    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.1f}, "
        f"{'성공' if info['is_success'] else '실패'}"
    )

env.close()
