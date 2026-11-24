from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

import ball_beam_env  # Ball Beam 환경 등록

MODEL_PATH = Path(__file__).with_name("ppo_ball_beam_alive")  # 이 파일과 같은 폴더에 저장

# 학습용 Environment 생성 (보상 A: 공이 살아있기만 하면 +1, 목표는 무시)
env = gym.make("BallBeamAlive-v0")

# PPO 모델 생성
model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

# 학습
model.learn(
    total_timesteps=200_000
)

# 학습 모델 저장
model.save(MODEL_PATH)

env.close()

print("PPO 학습 완료 (보상 A)")
