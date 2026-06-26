"""Dirichlet Flow Matching 可视化脚本.

生成 4 张图解释 Dirichlet FM：
  1. 2-class case: 线段上的流动（概率条形图 + 轨迹）
  2. 3-class case: 三角形 simplex 上多个 voxel 的流动轨迹
  3. Dirichlet(α) 的先验随 α 变化的直觉
  4. Gaussian FM vs Dirichlet FM 对比（为什么 Dirichlet 更干净）

输出到当前脚本同目录:
  dirichlet_fm_01_2class.png
  dirichlet_fm_02_3class_simplex.png
  dirichlet_fm_03_dirichlet_prior.png
  dirichlet_fm_04_gaussian_vs_dirichlet.png
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator

# 支持中文
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Serif CJK JP', 'DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

np.random.seed(0)
OUT_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------------ #
# Dirichlet FM 的条件路径 (Stark et al. 2024)
# 给定目标类 c, x_t | x_1=e_c ~ Dirichlet(1 + alpha(t) * e_c)
# alpha(t) = t / (1 - t + eps)  单调增, t=0 -> 0, t=1 -> inf
# ------------------------------------------------------------------ #

def alpha_schedule(t: float) -> float:
    return t / max(1e-4, 1.0 - t)


def sample_dirichlet_path(target_class: int, K: int, t: float, n: int = 1) -> np.ndarray:
    """Sample x_t from Dirichlet FM conditional path toward one-hot class target."""
    alpha = np.ones(K)
    alpha[target_class] += alpha_schedule(t)
    return np.random.dirichlet(alpha, size=n)


# ==================================================================== #
# Figure 1:  2-class case (segment)
# ==================================================================== #

def fig_2class():
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), gridspec_kw={'wspace': 0.35})
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    target = 1  # 真实类别 = 'part'

    for ax, t in zip(axes, ts):
        samples = sample_dirichlet_path(target, K=2, t=t, n=200)  # [n, 2]
        # 在 [0,1] 线段上画散点 (y 抖动)
        y_jitter = np.random.uniform(-0.08, 0.08, size=samples.shape[0])
        ax.scatter(samples[:, 1], y_jitter, s=12, alpha=0.45,
                   color='#1f77b4', edgecolors='none')

        # 目标 vertex
        ax.scatter([1.0], [0], s=180, marker='*', color='#d62728',
                   zorder=5, label='x_1 = e_part')

        # 均值
        mean = samples[:, 1].mean()
        ax.axvline(mean, color='#ff7f0e', ls='--', lw=1.5, alpha=0.7)
        ax.text(mean, 0.25, f'mean\n{mean:.2f}', ha='center', fontsize=8,
                color='#ff7f0e')

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.3, 0.3)
        ax.set_yticks([])
        ax.set_xlabel('P(part)')
        ax.set_title(f't = {t:.2f}', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # 端点标记
        if t == ts[0]:
            ax.text(0, -0.22, 'bg', ha='center', fontsize=9, color='#444')
            ax.text(1, -0.22, 'part', ha='center', fontsize=9, color='#444')

    fig.suptitle(
        "Dirichlet FM: 2-class case  |  x_t 始终在 [0,1] 内, 从均匀先验流向 vertex (one-hot)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / 'dirichlet_fm_01_2class.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


# ==================================================================== #
# Figure 2:  3-class case (triangle simplex) with multiple voxels
# ==================================================================== #

def simplex_to_2d(x: np.ndarray) -> np.ndarray:
    """(x0, x1, x2) on simplex -> 2D barycentric coords for plotting."""
    # 等边三角形顶点
    v0 = np.array([0.0, 0.0])      # class 0 (bg)
    v1 = np.array([1.0, 0.0])      # class 1 (part_A)
    v2 = np.array([0.5, np.sqrt(3) / 2])  # class 2 (part_B)
    V = np.stack([v0, v1, v2], axis=0)  # [3, 2]
    return x @ V


def draw_triangle(ax):
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    tri = plt.Polygon([v0, v1, v2], closed=True, fill=False,
                      ec='#333', lw=1.2)
    ax.add_patch(tri)
    # 顶点标签
    ax.text(v0[0] - 0.06, v0[1] - 0.06, 'bg\n(1,0,0)', fontsize=9,
            ha='right', va='top', color='#555')
    ax.text(v1[0] + 0.06, v1[1] - 0.06, 'part_A\n(0,1,0)', fontsize=9,
            ha='left', va='top', color='#555')
    ax.text(v2[0], v2[1] + 0.05, 'part_B\n(0,0,1)', fontsize=9,
            ha='center', va='bottom', color='#555')
    ax.set_aspect('equal')
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.2, 1.05)
    ax.axis('off')


def fig_3class_simplex():
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]

    # 3 个虚构 voxel，每个有不同的真实 class
    voxels = [
        {'label': 'voxel_a (真 bg)',    'target': 0, 'color': '#1f77b4'},
        {'label': 'voxel_b (真 part_A)', 'target': 1, 'color': '#2ca02c'},
        {'label': 'voxel_c (真 part_B)', 'target': 2, 'color': '#d62728'},
    ]

    for ax, t in zip(axes, ts):
        draw_triangle(ax)
        for v in voxels:
            samples = sample_dirichlet_path(v['target'], K=3, t=t, n=80)
            pts = simplex_to_2d(samples)
            ax.scatter(pts[:, 0], pts[:, 1], s=16, alpha=0.45,
                       color=v['color'], edgecolors='none',
                       label=v['label'] if t == ts[0] else None)
        ax.set_title(f't = {t:.2f}', fontsize=11)

    axes[0].legend(loc='upper left', bbox_to_anchor=(-0.1, -0.02),
                   fontsize=9, frameon=False, ncol=1)
    fig.suptitle(
        "Dirichlet FM: 3-class simplex (triangle)  |  3 个 voxel 各自流向自己的真实 vertex\n"
        "中间时刻 x_t 是合法概率分布 (sum=1, >=0), 天然表达模型置信度",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / 'dirichlet_fm_02_3class_simplex.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


# ==================================================================== #
# Figure 3:  Dirichlet(alpha) prior 直观图
# ==================================================================== #

def fig_dirichlet_prior():
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    alpha_configs = [
        (np.array([1.0, 1.0, 1.0]),    'Dirichlet(1,1,1)\n--> 均匀分布  (FM 起点 x_0)'),
        (np.array([2.0, 5.0, 2.0]),    'Dirichlet(2,5,2)\n--> 偏向 part_A  (中期)'),
        (np.array([1.0, 20.0, 1.0]),   'Dirichlet(1,20,1)\n--> 强烈指向 part_A  (后期)'),
        (np.array([1.0, 1e6, 1.0]),    'Dirichlet(1,inf,1)\n--> 退化到 vertex  (终点 x_1)'),
    ]
    for ax, (alpha, title) in zip(axes, alpha_configs):
        draw_triangle(ax)
        samples = np.random.dirichlet(alpha, size=600)
        pts = simplex_to_2d(samples)
        ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.45,
                   color='#8e44ad', edgecolors='none')
        ax.set_title(title, fontsize=10)

    fig.suptitle(
        "Dirichlet(alpha) 随 concentration 变化  |  FM 条件路径 alpha_t = 1 + alpha(t) * e_c 从左到右移动",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / 'dirichlet_fm_03_dirichlet_prior.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


# ==================================================================== #
# Figure 4:  Gaussian FM vs Dirichlet FM 对比 (2-class)
# ==================================================================== #

def fig_gaussian_vs_dirichlet():
    fig, (ax_g, ax_d) = plt.subplots(2, 1, figsize=(12, 5.5),
                                     gridspec_kw={'hspace': 0.55})

    # 两种 FM 在 2-class 下的轨迹
    # Gaussian FM: x ∈ R^2 (logits)，目标 x_1 = (0, 5) 表示 "class 1 logit 高"
    # 先不限制 x 范围
    ts = np.linspace(0, 1, 60)
    n_tracks = 40

    # -------- Gaussian FM --------
    x1_g = np.array([0.0, 5.0])  # 目标 logits
    for _ in range(n_tracks):
        x0 = np.random.randn(2)
        traj = np.array([(1 - t) * x0 + t * x1_g for t in ts])
        # 画 logit[1] (class 1 的 logit)
        ax_g.plot(ts, traj[:, 1], color='#bbb', lw=0.7, alpha=0.6)
    ax_g.axhline(5, color='#d62728', ls='--', lw=1.5,
                 label='x_1 target = 5 (logit of class 1)')
    ax_g.axhline(0, color='#ccc', lw=0.5)
    ax_g.fill_between([0, 1], [0, 0], [5, 5], color='#2ca02c', alpha=0.05)
    ax_g.text(0.02, 5.5, '[X] x_t 可以飞到任意实数范围',
              color='#d62728', fontsize=10)
    ax_g.text(0.02, -3.5, '[X] 最后要靠 softmax(logits) 才能得到概率',
              color='#d62728', fontsize=10)
    ax_g.set_xlim(0, 1)
    ax_g.set_ylim(-4, 9)
    ax_g.set_xlabel('t (flow time)')
    ax_g.set_ylabel('x_t[1] (class_1 logit)')
    ax_g.set_title('Gaussian FM on logits (SLAT 式)  |  x_t in R^2, 目标是一个任意点', fontsize=11)
    ax_g.legend(loc='upper left', fontsize=9)
    ax_g.grid(alpha=0.3)

    # -------- Dirichlet FM --------
    for _ in range(n_tracks):
        traj = []
        for t in ts:
            x_t = sample_dirichlet_path(target_class=1, K=2, t=t, n=1)[0]
            traj.append(x_t[1])
        ax_d.plot(ts, traj, color='#8e44ad', lw=0.7, alpha=0.5)
    ax_d.axhline(1.0, color='#2ca02c', ls='--', lw=1.5,
                 label='x_1 = e_1 (vertex, one-hot)')
    ax_d.axhline(0.5, color='#ccc', lw=0.5)
    ax_d.fill_between([0, 1], [0, 0], [1, 1], color='#2ca02c', alpha=0.1)
    ax_d.text(0.02, 0.02, '[OK] x_t 属于 [0,1], 全程合法概率',
              color='#2ca02c', fontsize=10)
    ax_d.text(0.02, 0.88, '[OK] 任意时刻 x_t 就是类别分布',
              color='#2ca02c', fontsize=10)
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(-0.05, 1.05)
    ax_d.set_xlabel('t (flow time)')
    ax_d.set_ylabel('x_t[1] = P(class_1)')
    ax_d.set_title('Dirichlet FM (simplex 内)  |  x_t in Delta^1 = [0,1], 目标是 simplex 顶点', fontsize=11)
    ax_d.legend(loc='lower right', fontsize=9)
    ax_d.grid(alpha=0.3)

    fig.suptitle(
        "为什么选 Dirichlet FM？  (40 条随机 voxel 轨迹, 目标类 = part)",
        fontsize=12, y=1.01,
    )
    out = OUT_DIR / 'dirichlet_fm_04_gaussian_vs_dirichlet.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


if __name__ == '__main__':
    print(f'Writing figures to: {OUT_DIR}')
    fig_2class()
    fig_3class_simplex()
    fig_dirichlet_prior()
    fig_gaussian_vs_dirichlet()
    print('done.')
