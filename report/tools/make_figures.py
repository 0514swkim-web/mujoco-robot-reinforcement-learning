"""보고서 그림 자동 생성.

  python make_figures.py curves    # 학습 곡선 + 평가 막대그래프 (logs/, results/ 필요)
  python make_figures.py renders   # 장면 캡처 · 학습 전/후 비교 · GIF (MUJOCO_GL=egl)
  python make_figures.py all

결과는 report/figures/ 에 저장.
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
FIG = REPORT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ---------- 스타일 (한글 폰트 + 차분한 축) ----------
for p in ["/mnt/c/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgun.ttf"]:
    if Path(p).exists():
        font_manager.fontManager.addfont(p)
        if Path(p.replace("malgun.ttf", "malgunbd.ttf")).exists():
            font_manager.fontManager.addfont(p.replace("malgun.ttf", "malgunbd.ttf"))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=p).get_name()
        break
plt.rcParams.update({
    "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 9.5,
    "figure.facecolor": "white", "axes.facecolor": "white", "lines.linewidth": 2,
})
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"

LABEL = {
    "ball_beam": "Ball & Beam (보상 B: 목표 위치)",
    "ball_beam_alive": "Ball & Beam (보상 A: 살아있기만)",
    "diff_drive": "Diff Drive (목표 지점 주행)",
    "arm_dense": "2-Link Arm · Dense SAC",
    "arm_sparse": "2-Link Arm · Sparse SAC (HER 없음)",
    "arm_sparse_her": "2-Link Arm · Sparse SAC + HER",
}


def load_monitor(job):
    f = REPORT / "logs" / job / "monitor.monitor.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, skiprows=1)
    df["timesteps"] = df["l"].cumsum()
    return df


def smooth(s, frac=0.02, min_w=10):
    w = max(min_w, int(len(s) * frac))
    return s.rolling(w, min_periods=1).mean()


def k_fmt(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))


# ---------- 1. 예제별 학습 곡선 (보상 / 성공률 두 패널, 축 하나씩) ----------
def fig_learning(job, color=BLUE):
    df = load_monitor(job)
    if df is None:
        print("skip (no log):", job); return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    ax.plot(df["timesteps"], df["r"], color=color, alpha=0.15, linewidth=1)
    ax.plot(df["timesteps"], smooth(df["r"]), color=color)
    ax.set_title("에피소드 보상"); ax.set_xlabel("학습 스텝"); ax.set_ylabel("보상 합")
    k_fmt(ax)
    ax = axes[1]
    ax.plot(df["timesteps"], smooth(df["is_success"]) * 100, color=color)
    ax.set_ylim(-3, 103)
    ax.set_title("성공률 (이동 평균)"); ax.set_xlabel("학습 스텝"); ax.set_ylabel("%")
    k_fmt(ax)
    fig.suptitle(LABEL[job], fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / f"curve_{job}.png"); plt.close(fig)


# ---------- 2. 비교 실험: 같은 지표(성공률)로 겹쳐 그림 ----------
def fig_compare(jobs, colors, title, fname):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    drew = False
    for job, c in zip(jobs, colors):
        df = load_monitor(job)
        if df is None:
            continue
        y = smooth(df["is_success"]) * 100
        ax.plot(df["timesteps"], y, color=c, label=LABEL[job])
        ax.annotate(f"{y.iloc[-1]:.0f}%", (df["timesteps"].iloc[-1], y.iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", color="#52514e")
        drew = True
    if not drew:
        plt.close(fig); return
    ax.set_ylim(-3, 108); ax.set_xlabel("학습 스텝"); ax.set_ylabel("성공률 (%)")
    ax.set_title(title); k_fmt(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / fname); plt.close(fig)


# ---------- 3. 평가 결과: 무작위 vs 직접 학습 vs 수업 배포 모델 ----------
def load_results():
    out = {}
    for f in sorted((REPORT / "results").glob("*.json")):
        out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return out


def fig_eval_bars(res):
    jobs = [j for j in LABEL if j in res]
    if not jobs:
        return
    x = np.arange(len(jobs)); w = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    series = [("random", "무작위 행동", GRAY), ("mine", "직접 학습", BLUE),
              ("course_pretrained", "수업 배포 모델", ORANGE)]
    for i, (k, name, c) in enumerate(series):
        vals = [res[j][k]["success_rate"] * 100 if k in res[j] else 0 for j in jobs]
        bars = ax.bar(x + (i - 1) * w, vals, w - 0.03, color=c, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center",
                    fontsize=7.5, color="#52514e")
    ax.set_xticks(x, [LABEL[j].replace(" (", "\n(").replace(" · ", "\n") for j in jobs], fontsize=8)
    ax.set_ylim(0, 112); ax.set_ylabel("성공률 (%)")
    ax.set_title(f"평가 성공률 (각 {res[jobs[0]]['mine']['n']} 에피소드, deterministic)", pad=28)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "eval_success.png"); plt.close(fig)


def write_results_md(res):
    rows = ["| 실험 | 알고리즘 | 학습 스텝 | 학습 시간 | 무작위 성공률 | 직접 학습 성공률 | 수업 모델 성공률 | 직접 학습 평균 보상 | 최종 거리(cm) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for j in LABEL:
        if j not in res:
            continue
        r = res[j]
        course = f"{r['course_pretrained']['success_rate']*100:.0f}%" if "course_pretrained" in r else "-"
        rows.append(f"| {LABEL[j]} | {r['algo']} | {r['timesteps']:,} | {r['train_seconds']/60:.1f}분 | "
                    f"{r['random']['success_rate']*100:.0f}% | **{r['mine']['success_rate']*100:.0f}%** | "
                    f"{course} | {r['mine']['return_mean']:.1f} | "
                    f"{r['mine']['final_dist_mean']*100:.1f} |")
    (REPORT / "results" / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


def curves():
    fig_learning("ball_beam", BLUE)
    fig_learning("ball_beam_alive", ORANGE)
    fig_learning("diff_drive", BLUE)
    fig_learning("arm_dense", BLUE)
    fig_learning("arm_sparse", ORANGE)
    fig_learning("arm_sparse_her", AQUA)
    fig_compare(["ball_beam_alive", "ball_beam"], [ORANGE, BLUE],
                "보상 설계 비교: Ball & Beam 성공률", "compare_ball_beam_reward.png")
    fig_compare(["arm_dense", "arm_sparse", "arm_sparse_her"], [BLUE, ORANGE, AQUA],
                "Dense vs Sparse vs Sparse+HER: 2-Link Arm 성공률", "compare_arm_her.png")
    res = load_results()
    fig_eval_bars(res)
    if res:
        write_results_md(res)


# =====================================================================
# 렌더링
# =====================================================================
def renders():
    import mujoco
    import imageio.v2 as imageio
    import gymnasium as gym
    from stable_baselines3 import PPO, SAC
    import ball_beam_env, two_link_arm_env, diff_drive_env  # noqa: F401

    ROBOTS = {
        # name: (env id, algo, 내 모델, 수업 모델, 환경 모듈, seed)
        "ball_beam":  ("BallBeam-v0", PPO, "ball_beam", "ppo_ball_beam", ball_beam_env, 7),
        "arm":        ("TwoLinkReachSparse-v0", SAC, "arm_sparse_her", "sac_her_two_link_arm", two_link_arm_env, 3),
        "diff_drive": ("DiffDriveGoal-v0", PPO, "diff_drive", "ppo_diff_drive", diff_drive_env, 5),
    }
    TITLE = {"ball_beam": "Ball & Beam", "arm": "2-Link Arm (SAC+HER)", "diff_drive": "Diff Drive"}

    def load_model(env_id, algo, mine, course):
        p = REPORT / "models" / f"{mine}.zip"
        src = p if p.exists() else ROOT / f"{course}.zip"
        env = gym.make(env_id) if "Sparse" in env_id else None
        return algo.load(src, env=env), ("직접 학습" if p.exists() else "수업 배포 모델")

    def rollout(env_id, policy, seed):
        env = gym.make(env_id, render_mode="rgb_array", width=640, height=480)
        env.action_space.seed(seed)
        obs, _ = env.reset(seed=seed)
        frames, done, info = [env.render()], False, {}
        while not done:
            a = env.action_space.sample() if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, r, term, trunc, info = env.step(a)
            frames.append(env.render()); done = term or trunc
        env.close()
        return frames, bool(info.get("is_success", 0))

    for name, (env_id, algo, mine, course, mod, seed) in ROBOTS.items():
        model, src = load_model(env_id, algo, mine, course)
        fps = int(gym.make(env_id).unwrapped.metadata["render_fps"])
        f_rand, ok_rand = rollout(env_id, None, seed)
        f_rl, ok_rl = rollout(env_id, model, seed)

        # GIF
        step = max(1, fps // 15)
        imageio.mimsave(FIG / f"{name}_before_random.gif", f_rand[::step], duration=step / fps, loop=0)
        imageio.mimsave(FIG / f"{name}_after_trained.gif", f_rl[::step], duration=step / fps, loop=0)

        # 같은 시점 4장씩 비교 (같은 seed → 같은 시작 조건). 시간축은 더 긴 쪽 기준,
        # 먼저 끝난 에피소드(예: 공이 떨어짐)는 마지막 장면 + "종료" 표시
        n = max(len(f_rand), len(f_rl))
        idx = [0, n // 8, n // 4, n // 2, n - 1]
        fig, axes = plt.subplots(2, 5, figsize=(14, 4.4))
        rows = [(f_rand, "학습 전 (무작위)", ok_rand), (f_rl, "학습 후", ok_rl)]
        for row, (frames, lab, ok) in enumerate(rows):
            for col, i in enumerate(idx):
                ax = axes[row, col]
                ended = i >= len(frames)
                ax.imshow(frames[min(i, len(frames) - 1)]); ax.set_xticks([]); ax.set_yticks([])
                ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(False)
                if row == 0:
                    ax.set_title(f"t = {i / fps:.1f}s", fontsize=14, fontweight="normal")
                if ended:
                    ax.text(0.97, 0.05, f"{(len(frames) - 1) / fps:.1f}s에 종료", transform=ax.transAxes,
                            ha="right", fontsize=12, color="white",
                            bbox=dict(boxstyle="round,pad=0.3", fc="#0b0b0b", alpha=0.6, lw=0))
                if col == 0:
                    ax.text(0.02, 0.95, f"{lab} → {'성공' if ok else '실패'}", transform=ax.transAxes,
                            va="top", fontsize=13, color="white",
                            bbox=dict(boxstyle="round,pad=0.3", fc="#0b0b0b", alpha=0.6, lw=0))
        fig.tight_layout(); fig.savefig(FIG / f"{name}_before_after.png"); plt.close(fig)
        print("rendered", name, ok_rand, ok_rl)

    # ---------- 장면 시각화: 관절축 · 접촉점/접촉력 · 원본 export vs scene ----------
    def snapshot(xml, cam, fname, flags=(), settle=0, setup=None, w=960, h=720, size=0.03):
        m = mujoco.MjModel.from_xml_path(str(xml))
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
        m.vis.global_.offheight = max(m.vis.global_.offheight, h)
        m.stat.meansize = size              # 관절 화살표·접촉 표시 크기 기준 (작은 로봇에 맞춤)
        m.vis.quality.shadowsize = 8192     # 그림자 해상도 ↑ (바닥 얼룩 방지)
        d = mujoco.MjData(m)
        if setup:
            setup(m, d)
        mujoco.mj_forward(m, d)
        for _ in range(settle):
            mujoco.mj_step(m, d)
        r = mujoco.Renderer(m, h, w)
        c = mujoco.MjvCamera()
        c.distance, c.azimuth, c.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
        c.lookat[:] = cam["lookat"]
        opt = mujoco.MjvOption()
        for f in flags:
            opt.flags[f] = 1
        m.vis.scale.jointlength = 2.0; m.vis.scale.jointwidth = 0.08
        m.vis.scale.contactwidth = 0.3; m.vis.scale.contactheight = 0.1
        m.vis.scale.forcewidth = 0.1
        # 가장 큰 접촉력 화살표가 cam 거리의 약 15% 길이가 되도록 배율 자동 조정
        fmax, f6 = 0.0, np.zeros(6)
        for i in range(d.ncon):
            mujoco.mj_contactForce(m, d, i, f6); fmax = max(fmax, abs(f6[0]))
        if fmax > 0:
            m.vis.map.force = 0.15 * cam["distance"] * m.stat.meanmass / fmax   # MuJoCo 가 meanmass 로 나눔
        r.update_scene(d, camera=c, scene_option=opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0     # 오프스크린 그림자 얼룩 제거
        img = r.render(); r.close()
        imageio.imwrite(FIG / fname, img)

    V = mujoco.mjtVisFlag
    viz = (V.mjVIS_JOINT, V.mjVIS_CONTACTPOINT, V.mjVIS_CONTACTFORCE)
    oblique = lambda cam, az=135, el=-25: {**cam, "azimuth": az, "elevation": el}

    bb_cam = oblique(ball_beam_env.CAMERA, 120, -20)
    def bb_setup(m, d):
        q = m.joint("ball_joint").qposadr[0]
        d.qpos[q:q + 7] = [0.05, 0, 0.123 + 0.015, 1, 0, 0, 0]
    snapshot(ROOT / "ball_beam/ball_beam.xml", bb_cam, "scene_ball_beam_raw_export.png")
    snapshot(ROOT / "ball_beam/ball_beam_scene.xml", bb_cam, "scene_ball_beam_scene.png", setup=bb_setup, settle=20)
    snapshot(ROOT / "ball_beam/ball_beam_scene.xml", bb_cam, "scene_ball_beam_viz.png", viz, setup=bb_setup,
             settle=20, size=0.008)

    arm_cam = oblique(two_link_arm_env.CAMERA, 120, -35)
    snapshot(ROOT / "two_link_arm/two_link_arm.xml", arm_cam, "scene_arm_raw_export.png")
    snapshot(ROOT / "two_link_arm/two_link_arm_scene.xml", arm_cam, "scene_arm_scene.png")
    snapshot(ROOT / "two_link_arm/two_link_arm_scene.xml", arm_cam, "scene_arm_viz.png", viz)

    dd_cam = {"distance": 0.9, "azimuth": 135, "elevation": -25, "lookat": np.array([0, 0, 0.05])}
    snapshot(ROOT / "diff_drive/diff_drive.xml", dd_cam, "scene_diff_drive_raw_export.png")
    snapshot(ROOT / "diff_drive/diff_drive_scene.xml", dd_cam, "scene_diff_drive_scene.png", settle=300)
    snapshot(ROOT / "diff_drive/diff_drive_scene.xml", dd_cam, "scene_diff_drive_viz.png", viz, settle=300)
    print("scene snapshots done")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("curves", "all"):
        curves()
    if what in ("renders", "all"):
        renders()
