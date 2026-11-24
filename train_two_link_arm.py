from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC

import two_link_arm_env  # 2관절 로봇팔 환경 등록

MODEL_PATH = Path(__file__).with_name("sac_two_link_arm")  # 이 파일과 같은 폴더에 저장

# 학습용 Environment 생성 (촘촘한 보상: 가까울수록 좋음)
env = gym.make("TwoLinkReach-v0")

# SAC 모델 생성
model = SAC(
    "MlpPolicy",
    env,
    gamma=0.95,                             # 2초짜리 짧은 과제 → 가까운 미래 보상에 집중
    policy_kwargs=dict(net_arch=[64, 64]),  # 작은 신경망 → CPU 에서도 빨리 학습
    verbose=1
)

# 학습
model.learn(
    total_timesteps=30_000
)

# 학습 모델 저장
model.save(MODEL_PATH)

env.close()

print("SAC 학습 완료")
