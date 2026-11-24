from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer

import two_link_arm_env  # 2관절 로봇팔 환경 등록

MODEL_PATH = Path(__file__).with_name("sac_her_two_link_arm")  # 이 파일과 같은 폴더에 저장

# 학습용 Environment 생성 (성공했을 때만 보상)
env = gym.make("TwoLinkReachSparse-v0")

# SAC + HER 모델 생성
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,        # HER: 실패한 경험을
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,                       # "그 위치가 목표였다면 성공"으로 4번씩 다시 씀
        goal_selection_strategy="future",       # 바꿀 목표 = 그 뒤에 실제로 도착한 곳
    ),
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

print("SAC + HER 학습 완료")
