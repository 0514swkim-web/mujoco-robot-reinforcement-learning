"""2관절 로봇팔 손끝 뻗기 (Reach) 환경 (MuJoCo + Gymnasium).

책상 위에 수평으로 도는 2관절 팔이 있습니다. 손끝(파란 점)을
무작위로 나타나는 빨간 목표점에 가져가 머무르면 성공입니다.
환경은 2가지입니다. 물리는 같고 "보상을 주는 방법"만 다릅니다.

[1] TwoLinkReach-v0  (촘촘한 보상, SAC)
    관찰 12개: [cos q1, sin q1, cos q2, sin q2, 관절속도 2개,
               손끝 x, y, 목표 x, y, (목표 - 손끝) x, y]
    보상 (매 스텝, 가중치는 아래 상수)
        - DIST_WEIGHT   x 손끝~목표 거리(m)    멀수록 벌점 → 조금만 가까워져도 보상이 좋아짐
        - ACTION_WEIGHT x (행동1^2 + 행동2^2)  힘을 많이 쓰면 벌점
        + SUCCESS_BONUS                       목표 2cm 안이면 보너스

[2] TwoLinkReachSparse-v0  (성공했을 때만 주는 보상, SAC + HER)
    관찰은 딕셔너리 3개 (HER 가 목표를 바꿔 끼워 다시 계산할 수 있게)
        observation   : [cos q1, sin q1, cos q2, sin q2, 관절속도 2개, 손끝 x, y]
        achieved_goal : 지금 손끝 위치 [x, y]
        desired_goal  : 목표 위치 [x, y]
    보상: 목표 2cm 안이면 0, 아니면 -1   (가까워져도 힌트가 없음 → HER 가 필요)

행동(action) 2개 (-1 ~ 1): 관절1, 관절2 모터 토크 (XML 의 ctrlrange ±0.2 N·m 로 바뀜)

종료: 따로 없음. 25 스텝(2초)이 지나면 truncated. 끝났을 때 목표 위에 머물러 있어야 함
성공 (info["is_success"]): 손끝이 목표 2cm 안
"""

import time
from pathlib import Path

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

XML_PATH = Path(__file__).parent / "two_link_arm" / "two_link_arm_scene.xml"   # 학생이 고친 복사본

N_SUBSTEPS = 40             # 행동 1번에 물리 스텝 40번 (0.002s x 40 = 0.08s, 초당 12.5번 제어)
TARGET_R_MIN = 0.08         # 목표는 팔이 닿는 고리 모양 영역 안: 중심에서 8cm ~
TARGET_R_MAX = 0.26         #                                     ~ 26cm
SUCCESS_DIST = 0.02         # 손끝이 목표 2cm 안이면 성공

# ---------- 보상 가중치 (촘촘한 보상 환경) ----------
DIST_WEIGHT = 5.0           # 거리(m) 벌점: 10cm 떨어져 있으면 매 스텝 -0.5
ACTION_WEIGHT = 0.05        # 행동 크기 벌점
SUCCESS_BONUS = 1.0         # 목표 2cm 안이면 매 스텝 +1

CAMERA = {"distance": 0.75, "azimuth": 90.0, "elevation": -60.0,
          "lookat": np.array([0.0, 0.0, 0.03])}


class TwoLinkReachEnv(gym.Env):
    """[1] 촘촘한 보상 (dense reward)"""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12.5}

    def __init__(self, render_mode=None, width=640, height=480):
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        m = self.model

        # 이름으로 찾기 (Fusion export 로 바꿔도 이름만 같으면 그대로 동작)
        self._qadr = np.array([m.joint(n).qposadr[0] for n in ("joint1", "joint2")])
        self._vadr = np.array([m.joint(n).dofadr[0] for n in ("joint1", "joint2")])
        self._motors = np.array([m.actuator(n).id for n in ("motor1", "motor2")])
        self._ctrl_low = m.actuator_ctrlrange[self._motors, 0]
        self._ctrl_high = m.actuator_ctrlrange[self._motors, 1]
        # 관절 축이 +z / -z 어느 쪽으로 내보내져도 "+ = 반시계 방향" 이 되도록 부호 맞춤
        self._axis_sign = np.sign([m.joint(n).axis[2] for n in ("joint1", "joint2")])
        self._tip = m.site("fingertip").id
        self._target = m.site("target").id

        self.goal = np.array([0.2, 0.0])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = self._make_obs_space()

        m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)      # 녹화 해상도 여유
        m.vis.global_.offheight = max(m.vis.global_.offheight, height)
        self.mujoco_renderer = MujocoRenderer(m, self.data, default_cam_config=CAMERA,
                                              width=width, height=height)
        self._last_frame = None

    def _make_obs_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float64)

    # ---------- 관찰 ----------
    def _tip_xy(self):
        return self.data.site_xpos[self._tip][:2].copy()

    def _robot_state(self):
        q1, q2 = self._axis_sign * self.data.qpos[self._qadr]
        qd1, qd2 = self._axis_sign * self.data.qvel[self._vadr]
        tip_x, tip_y = self._tip_xy()
        return np.array([np.cos(q1), np.sin(q1), np.cos(q2), np.sin(q2), qd1, qd2, tip_x, tip_y])

    def _get_obs(self):
        tip = self._tip_xy()
        return np.concatenate([self._robot_state(), self.goal, self.goal - tip])

    # ---------- 보상 ----------
    def _reward(self, dist, action):
        r_dist = -DIST_WEIGHT * dist
        r_action = -ACTION_WEIGHT * float(np.sum(action ** 2))
        r_success = SUCCESS_BONUS if dist < SUCCESS_DIST else 0.0
        return r_dist + r_action + r_success

    # ---------- reset / step ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        m, d, rng = self.model, self.data, self.np_random
        mujoco.mj_resetData(m, d)

        # 팔: 무작위 자세에서 시작
        d.qpos[self._qadr] = [rng.uniform(-np.pi, np.pi), rng.uniform(-2.5, 2.5)]   # joint2 는 ±150도 제한

        # 목표: 팔이 닿는 고리 영역 안 무작위 점 (넓이에 고르게)
        r = np.sqrt(rng.uniform(TARGET_R_MIN ** 2, TARGET_R_MAX ** 2))
        th = rng.uniform(-np.pi, np.pi)
        self.goal = np.array([r * np.cos(th), r * np.sin(th)])
        m.site_pos[self._target][:2] = self.goal

        mujoco.mj_forward(m, d)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        d = self.data
        # 행동 -1~1 을 모터 토크(ctrlrange)로 바꿈
        torque = self._ctrl_low + (a + 1.0) / 2.0 * (self._ctrl_high - self._ctrl_low)
        d.ctrl[self._motors] = self._axis_sign * torque
        mujoco.mj_step(self.model, d, nstep=N_SUBSTEPS)

        obs = self._get_obs()
        dist = float(np.linalg.norm(self._tip_xy() - self.goal))
        reward = self._reward(dist, a)
        info = {"is_success": float(dist < SUCCESS_DIST), "dist": dist}
        if self.render_mode == "human":
            self.render()
        return obs, reward, False, False, info

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


class TwoLinkReachSparseEnv(TwoLinkReachEnv):
    """[2] 성공했을 때만 주는 보상 (sparse reward) + HER 용 딕셔너리 관찰"""

    def _make_obs_space(self):
        return spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float64),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64),
        })

    def _get_obs(self):
        return {
            "observation": self._robot_state(),
            "achieved_goal": self._tip_xy(),      # 지금 손끝이 있는 곳
            "desired_goal": self.goal.copy(),     # 가야 할 곳
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """목표 2cm 안이면 0, 아니면 -1 (HER 가 목표를 바꿔 다시 계산할 수 있도록 배열도 처리)."""
        dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(dist >= SUCCESS_DIST).astype(np.float32)

    def _reward(self, dist, action):
        return float(self.compute_reward(self._tip_xy(), self.goal, None))


if "TwoLinkReach-v0" not in gym.registry:
    gym.register(id="TwoLinkReach-v0", entry_point="two_link_arm_env:TwoLinkReachEnv",
                 max_episode_steps=25)
    gym.register(id="TwoLinkReachSparse-v0", entry_point="two_link_arm_env:TwoLinkReachSparseEnv",
                 max_episode_steps=25)
