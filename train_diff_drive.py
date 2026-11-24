from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

import diff_drive_env  # 2륜 이동로봇 환경 등록

MODEL_PATH = Path(__file__).with_name("ppo_diff_drive")  # 이 파일과 같은 폴더에 저장

# 학습용 Environment 생성
env = gym.make("DiffDriveGoal-v0")

# PPO 모델 생성
model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

# 학습
model.learn(
    total_timesteps=100_000
)

# 학습 모델 저장
model.save(MODEL_PATH)

env.close()

print("PPO 학습 완료")
