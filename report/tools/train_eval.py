"""보고서용 학습 + 평가 스크립트.

수업 train_*.py 와 같은 알고리즘·하이퍼파라미터로 다시 학습하되,
- Monitor 로 에피소드마다 보상/성공 여부를 CSV 로 남기고 (학습 곡선용)
- 모델은 report/models/ 에 저장 (원래 *.zip 은 건드리지 않음)
- 학습 후 무작위 정책 / 내가 학습한 정책 / 수업 배포 모델을 같은 조건으로 평가

사용법:  python train_eval.py <job>      (job 이름은 JOBS 참고)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[2]          # mujoco_test
REPORT = ROOT / "report"
sys.path.insert(0, str(ROOT))

import ball_beam_env      # noqa: E402,F401  환경 등록
import two_link_arm_env   # noqa: E402,F401
import diff_drive_env     # noqa: E402,F401

SAC_KW = dict(gamma=0.95, policy_kwargs=dict(net_arch=[64, 64]))
HER_KW = dict(replay_buffer_class=HerReplayBuffer,
              replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"))

# job: (환경 id, 알고리즘, policy, 추가 인자, 스텝 수, 수업 배포 모델 파일)
JOBS = {
    "ball_beam":        ("BallBeam-v0",           PPO, "MlpPolicy",        {},                   200_000, "ppo_ball_beam"),
    "ball_beam_alive":  ("BallBeamAlive-v0",      PPO, "MlpPolicy",        {},                   200_000, "ppo_ball_beam_alive"),
    "diff_drive":       ("DiffDriveGoal-v0",      PPO, "MlpPolicy",        {},                   100_000, "ppo_diff_drive"),
    "arm_dense":        ("TwoLinkReach-v0",       SAC, "MlpPolicy",        SAC_KW,                30_000, "sac_two_link_arm"),
    "arm_sparse":       ("TwoLinkReachSparse-v0", SAC, "MultiInputPolicy", SAC_KW,                30_000, "sac_sparse_two_link_arm"),
    "arm_sparse_her":   ("TwoLinkReachSparse-v0", SAC, "MultiInputPolicy", {**SAC_KW, **HER_KW}, 30_000, "sac_her_two_link_arm"),
}

N_EVAL = 100


def evaluate(env_id, policy=None, n=N_EVAL, seed=1000):
    """policy=None 이면 무작위 행동. 같은 seed 로 같은 시작 조건에서 비교."""
    env = gym.make(env_id)
    env.action_space.seed(seed)
    succ, rets, dists, lens = [], [], [], []
    for i in range(n):
        obs, info = env.reset(seed=seed + i)
        done, ret, steps = False, 0.0, 0
        while not done:
            if policy is None:
                action = env.action_space.sample()
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            ret += float(r)
            steps += 1
            done = term or trunc
        succ.append(float(info["is_success"]))
        rets.append(ret)
        dists.append(float(info.get("dist", np.nan)))
        lens.append(steps)
    env.close()
    return {"success_rate": float(np.mean(succ)), "return_mean": float(np.mean(rets)),
            "return_std": float(np.std(rets)), "final_dist_mean": float(np.nanmean(dists)),
            "ep_len_mean": float(np.mean(lens)), "n": n}


def main(job):
    torch.set_num_threads(2)
    env_id, algo, policy, kw, steps, course_model = JOBS[job]
    log_dir = REPORT / "logs" / job
    log_dir.mkdir(parents=True, exist_ok=True)
    (REPORT / "models").mkdir(exist_ok=True)
    (REPORT / "results").mkdir(exist_ok=True)

    env = Monitor(gym.make(env_id), str(log_dir / "monitor"), info_keywords=("is_success",))
    model = algo(policy, env, verbose=1, seed=0, **kw)
    t0 = time.time()
    model.learn(total_timesteps=steps)
    train_sec = time.time() - t0
    model.save(REPORT / "models" / f"{job}")
    env.close()

    load_env = gym.make(env_id) if "HER" in str(kw) or "replay_buffer_class" in kw else None
    mine = algo.load(REPORT / "models" / f"{job}", env=load_env)

    result = {
        "job": job, "env_id": env_id, "algo": algo.__name__, "timesteps": steps,
        "train_seconds": round(train_sec, 1),
        "random": evaluate(env_id, None),
        "mine": evaluate(env_id, mine),
    }
    # 수업 배포 모델은 저장소에 없으므로, 파일이 있을 때만 비교
    if (ROOT / f"{course_model}.zip").exists():
        course = algo.load(ROOT / course_model, env=load_env)
        result["course_pretrained"] = evaluate(env_id, course)
    (REPORT / "results" / f"{job}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
