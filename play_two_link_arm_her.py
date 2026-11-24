from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC

import two_link_arm_env  # 2관절 로봇팔 환경 등록

# HER 없이 학습한 모델을 보려면 "sac_sparse_two_link_arm" 으로 바꾸기
MODEL_PATH = Path(__file__).with_name("sac_her_two_link_arm")

env = gym.make(
    "TwoLinkReachSparse-v0",
    render_mode="human"
)

model = SAC.load(MODEL_PATH, env=env)  # HER 모델은 env 를 같이 넘겨야 함

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
        f"Reward = {total_reward:.0f}, "
        f"{'성공' if info['is_success'] else '실패'}"
    )

env.close()
