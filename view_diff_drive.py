import gymnasium as gym

import diff_drive_env  # 2륜 이동로봇 환경 등록

env = gym.make(
    "DiffDriveGoal-v0",
    render_mode="human"
)

observation, info = env.reset()

print("Observation:", observation)
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

try:
    while True:
        # Random Action
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        # Episode 종료 시 새로운 Episode 시작
        if terminated or truncated:
            observation, info = env.reset()

except KeyboardInterrupt:
    pass

finally:
    env.close()
