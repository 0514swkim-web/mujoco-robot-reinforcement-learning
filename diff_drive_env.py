"""2륜 이동로봇 목표 지점 주행 (Differential Drive) 환경 (MuJoCo + Gymnasium).

바퀴 2개 + 뒤쪽 볼캐스터로 굴러가는 로봇이, 바닥의 초록 원판(목표)까지
스스로 운전해서 들어가면 성공입니다. 목표는 매번 1.5m 안 무작위 위치입니다.

관찰(observation) 7개 — 모두 "로봇 기준" 좌표 (로봇 앞 = +x, 왼쪽 = +y)
    [목표 x, 목표 y, 목표까지 거리, cos(목표 방향), sin(목표 방향),
     앞으로 가는 속도, 회전 속도]
    로봇 기준이라서 로봇이 어디를 보고 있든 같은 규칙으로 운전할 수 있음

행동(action) 2개 (-1 ~ 1)
    [왼쪽 바퀴 속도, 오른쪽 바퀴 속도]   +1 = 앞으로 최고 속도(15 rad/s, 약 0.6 m/s)
    둘이 같으면 직진, 다르면 회전

보상(reward) — 매 스텝 아래 항목을 더함 (가중치는 아래 상수)
    + PROGRESS_WEIGHT x (이전 거리 - 지금 거리)   목표에 가까워진 만큼 +, 멀어지면 -
    - TIME_PENALTY                               매 스텝 조금씩 벌점 → 빨리 도착할수록 이득
    + ARRIVAL_BONUS                              목표 원판(반지름 10cm) 안에 들어오면 보너스
    "가까워진 만큼" 주는 보상이라 제자리에서 빙빙 돌거나 원을 그려도 이득이 없음

종료
    terminated : 목표에 도착하면
    truncated  : 200 스텝(10초)이 지나면 (gym.register 의 max_episode_steps)

성공 (info["is_success"]): 목표 원판 안에 도착
"""

import time
from pathlib import Path

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

XML_PATH = Path(__file__).parent / "diff_drive" / "diff_drive_scene.xml"   # 학생이 고친 복사본

N_SUBSTEPS = 25             # 행동 1번에 물리 스텝 25번 (0.002s x 25 = 0.05s, 초당 20번 제어)
GOAL_DIST_MIN = 0.3         # 목표는 시작점에서 0.3m ~
GOAL_DIST_MAX = 1.5         #                     ~ 1.5m 떨어진 무작위 위치
GOAL_RADIUS = 0.1           # 목표 원판 반지름 = 도착 판정 거리 (m)

# ---------- 보상 가중치 (수업에서 바꿔 보는 부분) ----------
PROGRESS_WEIGHT = 10.0      # 목표에 1m 가까워지면 +10
TIME_PENALTY = 0.05         # 매 스텝 -0.05 (10초 다 쓰면 -10)
ARRIVAL_BONUS = 10.0        # 도착하면 +10

CAMERA = {"distance": 3.6, "azimuth": 90.0, "elevation": -70.0,
          "lookat": np.array([0.0, 0.0, 0.0])}


class DiffDriveGoalEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, width=640, height=480):
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        m = self.model

        # 이름으로 찾기 (Fusion export 로 바꿔도 이름만 같으면 그대로 동작)
        self._base = m.body("base_link").id
        root = m.joint("root")
        self._root_qadr = root.qposadr[0]
        self._root_vadr = root.dofadr[0]
        self._motors = np.array([m.actuator(n).id for n in ("left_motor", "right_motor")])
        self._ctrl_low = m.actuator_ctrlrange[self._motors, 0]
        self._ctrl_high = m.actuator_ctrlrange[self._motors, 1]
        # 바퀴 축이 -y 로 내보내졌어도 "+ = 앞으로" 가 되도록 부호 맞춤
        self._wheel_dir = np.array([np.sign(m.joint(n).axis[1]) for n in ("left_wheel_joint", "right_wheel_joint")])
        self._goal_site = m.site("goal").id

        self.goal = np.array([1.0, 0.0])
        self.prev_dist = 1.0
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64)

        m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)      # 녹화 해상도 여유
        m.vis.global_.offheight = max(m.vis.global_.offheight, height)
        self.mujoco_renderer = MujocoRenderer(m, self.data, default_cam_config=CAMERA,
                                              width=width, height=height)
        self._last_frame = None

    # ---------- 관찰: 목표를 로봇 기준 좌표로 ----------
    def _goal_in_robot_frame(self):
        d = self.data
        R = d.xmat[self._base].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])                   # 로봇이 바라보는 방향
        dx, dy = self.goal - d.xpos[self._base][:2]
        gx = np.cos(yaw) * dx + np.sin(yaw) * dy             # 로봇 앞쪽 성분
        gy = -np.sin(yaw) * dx + np.cos(yaw) * dy            # 로봇 왼쪽 성분
        return gx, gy

    def _get_obs(self):
        d = self.data
        gx, gy = self._goal_in_robot_frame()
        dist = np.hypot(gx, gy)
        R = d.xmat[self._base].reshape(3, 3)
        v_fwd = R[:, 0] @ d.qvel[self._root_vadr:self._root_vadr + 3]   # 앞으로 가는 속도
        yaw_rate = d.qvel[self._root_vadr + 5]                         # 회전 속도
        return np.array([gx, gy, dist, gx / (dist + 1e-6), gy / (dist + 1e-6), v_fwd, yaw_rate])

    # ---------- reset / step ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        m, d, rng = self.model, self.data, self.np_random
        mujoco.mj_resetData(m, d)

        # 로봇: 원점에서 무작위 방향을 보고 시작
        yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[self._root_qadr:self._root_qadr + 7] = [0, 0, 0.001, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

        # 목표: 0.3m ~ 1.5m 떨어진 무작위 위치 (초록 원판을 그 위치로 옮김)
        r = rng.uniform(GOAL_DIST_MIN, GOAL_DIST_MAX)
        th = rng.uniform(-np.pi, np.pi)
        self.goal = np.array([r * np.cos(th), r * np.sin(th)])
        m.site_pos[self._goal_site][:2] = self.goal

        mujoco.mj_forward(m, d)
        self.prev_dist = float(np.linalg.norm(self.goal - d.xpos[self._base][:2]))
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        d = self.data
        # 행동 -1~1 을 바퀴 목표 속도(ctrlrange, ±15 rad/s)로 바꿈
        speed = self._ctrl_low + (a + 1.0) / 2.0 * (self._ctrl_high - self._ctrl_low)
        d.ctrl[self._motors] = speed * self._wheel_dir
        mujoco.mj_step(self.model, d, nstep=N_SUBSTEPS)

        dist = float(np.linalg.norm(self.goal - d.xpos[self._base][:2]))
        arrived = dist < GOAL_RADIUS

        # ----- 보상 -----
        r_progress = PROGRESS_WEIGHT * (self.prev_dist - dist)
        r_time = -TIME_PENALTY
        r_arrival = ARRIVAL_BONUS if arrived else 0.0
        reward = r_progress + r_time + r_arrival
        self.prev_dist = dist

        info = {"is_success": float(arrived), "dist": dist}
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, arrived, False, info

    def render(self):
        frame = self.mujoco_renderer.render(self.render_mode)
        if self.render_mode == "human":      # 창 모드에서는 실제 시간 속도로 재생
            dt = 1.0 / self.metadata["render_fps"]
            if self._last_frame is not None:
                time.sleep(max(0.0, dt - (time.perf_counter() - self._last_frame)))
            self._last_frame = time.perf_counter()
        return frame

    def close(self):
        self.mujoco_renderer.close()


if "DiffDriveGoal-v0" not in gym.registry:
    gym.register(id="DiffDriveGoal-v0", entry_point="diff_drive_env:DiffDriveGoalEnv",
                 max_episode_steps=200)
