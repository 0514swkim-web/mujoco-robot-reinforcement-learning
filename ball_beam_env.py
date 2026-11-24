"""볼 밸런싱 (Ball and Beam) 환경 (MuJoCo + Gymnasium).

기둥 위에 놓인 긴 판(beam)을 기울여서, 판 위를 굴러다니는 공을
초록색 목표 표시(target) 위에 멈춰 세우면 성공입니다.

관찰(observation) 5개
    [공 위치 x, 공 속도, beam 각도, beam 각속도, 목표까지 남은 거리(목표 x - 공 x)]
    공 위치·속도는 beam 을 따라 잰 값 (m, m/s), 각도는 rad

행동(action) 1개  (-1 ~ 1)
    beam 목표 각도.  -1 = -15도,  +1 = +15도  (XML 의 position 모터가 그 각도로 돌림)

보상(reward) — 매 스텝 아래 항목을 더함 (가중치는 아래 상수)
    + ALIVE_BONUS                     공이 beam 위에 있으면 +1
    - DIST_WEIGHT   x |목표 - 공|       목표에서 멀수록 벌점
    - VEL_WEIGHT    x |공 속도|         공이 빨리 굴러다니면 벌점
    - ACTION_WEIGHT x 행동^2            beam 을 크게 흔들면 벌점
    ※ reward_mode="alive" (BallBeamAlive-v0) 는 "보상 A": 살아있기만 하면 +1, 목표는 무시

종료
    terminated : 공이 beam 끝을 넘어가 떨어지면
    truncated  : 250 스텝(5초)이 지나면 (gym.register 의 max_episode_steps)

성공 (info["is_success"])
    공이 목표 2cm 안에 있고, 속도가 5cm/s 보다 느리면
"""

import time
from pathlib import Path

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

XML_PATH = Path(__file__).parent / "ball_beam" / "ball_beam_scene.xml"   # 학생이 고친 복사본

N_SUBSTEPS = 10             # 행동 1번에 물리 스텝 10번 (0.002s x 10 = 0.02s, 초당 50번 제어)
BEAM_HALF = 0.2             # beam 길이의 반 (m). 공이 이 밖으로 나가면 떨어진 것
BALL_RADIUS = 0.015         # 공 반지름 (m)
BEAM_TOP = 0.123            # beam 윗면 높이 (m)
START_RANGE = 0.15          # 공 시작 위치: -15cm ~ +15cm 에서 무작위
TARGET_RANGE = 0.15         # 목표 위치:   -15cm ~ +15cm 에서 무작위
SUCCESS_DIST = 0.02         # 목표 2cm 안이면 위치 조건 만족
SUCCESS_VEL = 0.05          # 공 속도가 5cm/s 보다 느리면 "멈춤"

# ---------- 보상 가중치 (수업에서 바꿔 보는 부분) ----------
ALIVE_BONUS = 1.0           # 공이 beam 위에 있으면 매 스텝 +1
DIST_WEIGHT = 5.0           # 목표까지 거리(m) 벌점: 10cm 떨어져 있으면 매 스텝 -0.5
VEL_WEIGHT = 0.5            # 공 속도(m/s) 벌점: 0.2 m/s 로 굴러가면 매 스텝 -0.1
ACTION_WEIGHT = 0.05        # 행동 크기 벌점

CAMERA = {"distance": 0.45, "azimuth": 90.0, "elevation": -35.0,
          "lookat": np.array([0.0, 0.0, 0.08])}


class BallBeamEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, reward_mode="target", width=640, height=480):
        self.render_mode = render_mode
        self.reward_mode = reward_mode          # "target" (기본) 또는 "alive" (보상 A)

        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        m = self.model

        # 이름으로 찾기 (Fusion export 로 바꿔도 이름만 같으면 그대로 동작)
        beam_joint = m.joint("beam_joint")
        self._beam_jid = beam_joint.id
        self._beam_body = beam_joint.bodyid[0]          # beam_joint 가 달린 body (= beam)
        self._beam_qadr = beam_joint.qposadr[0]
        self._beam_vadr = beam_joint.dofadr[0]
        ball_joint = m.joint("ball_joint")
        self._ball_qadr = ball_joint.qposadr[0]
        self._ball_vadr = ball_joint.dofadr[0]
        self._target = m.site("target").id
        self._motor = m.actuator("beam_motor").id
        self._ctrl_low, self._ctrl_high = m.actuator_ctrlrange[self._motor]
        # 관절 축이 +y / -y 어느 쪽으로 내보내져도 "+ 각도 = 오른쪽(+x) 끝이 내려감" 이 되도록 부호 맞춤
        self._axis_sign = np.sign(m.jnt_axis[self._beam_jid][1])

        self.target_x = 0.0
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float64)

        m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)      # 녹화 해상도 여유
        m.vis.global_.offheight = max(m.vis.global_.offheight, height)
        self.mujoco_renderer = MujocoRenderer(m, self.data, default_cam_config=CAMERA,
                                              width=width, height=height)
        self._last_frame = None

    # ---------- 공 위치: beam 을 따라 잰 x, 속도 ----------
    def _ball_on_beam(self):
        d = self.data
        beam_dir = d.xmat[self._beam_body].reshape(3, 3)[:, 0]          # beam 길이 방향 (월드 좌표)
        rel = d.qpos[self._ball_qadr:self._ball_qadr + 3] - d.xanchor[self._beam_jid]
        x = float(beam_dir @ rel)
        v = float(beam_dir @ d.qvel[self._ball_vadr:self._ball_vadr + 3])
        return x, v

    def _get_obs(self):
        x, v = self._ball_on_beam()
        angle = self._axis_sign * self.data.qpos[self._beam_qadr]
        angle_vel = self._axis_sign * self.data.qvel[self._beam_vadr]
        return np.array([x, v, angle, angle_vel, self.target_x - x])

    # ---------- reset / step ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        m, d, rng = self.model, self.data, self.np_random
        mujoco.mj_resetData(m, d)

        # 공: beam 위 무작위 위치에 정지 상태로 놓기
        x0 = rng.uniform(-START_RANGE, START_RANGE)
        d.qpos[self._ball_qadr:self._ball_qadr + 7] = [x0, 0.0, BEAM_TOP + BALL_RADIUS, 1, 0, 0, 0]

        # 목표: beam 위 무작위 위치 (초록 표시를 그 위치로 옮김)
        self.target_x = rng.uniform(-TARGET_RANGE, TARGET_RANGE)
        m.site_pos[self._target][0] = self.target_x

        mujoco.mj_forward(m, d)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        a = float(np.clip(action, -1.0, 1.0)[0])
        d = self.data
        # 행동 -1~1 을 모터 목표 각도(ctrlrange, ±15도)로 바꿈
        target_angle = self._ctrl_low + (a + 1.0) / 2.0 * (self._ctrl_high - self._ctrl_low)
        d.ctrl[self._motor] = self._axis_sign * target_angle
        mujoco.mj_step(self.model, d, nstep=N_SUBSTEPS)

        x, v = self._ball_on_beam()
        dist = abs(self.target_x - x)
        fell = abs(x) > BEAM_HALF                      # beam 끝을 넘어감 = 떨어짐

        # ----- 보상 -----
        r_alive = ALIVE_BONUS
        r_dist = -DIST_WEIGHT * dist
        r_vel = -VEL_WEIGHT * abs(v)
        r_action = -ACTION_WEIGHT * a ** 2
        if self.reward_mode == "alive":                # 보상 A: 살아있기만 하면 +1
            reward = r_alive
        else:                                          # 보상 B: 목표 위치까지 고려
            reward = r_alive + r_dist + r_vel + r_action

        success = (not fell) and dist < SUCCESS_DIST and abs(v) < SUCCESS_VEL
        info = {"is_success": float(success), "dist": dist}
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, fell, False, info

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


if "BallBeam-v0" not in gym.registry:
    gym.register(id="BallBeam-v0", entry_point="ball_beam_env:BallBeamEnv",
                 max_episode_steps=250)
    gym.register(id="BallBeamAlive-v0", entry_point="ball_beam_env:BallBeamEnv",
                 max_episode_steps=250, kwargs={"reward_mode": "alive"})
