"""高维 K 下 Dirichlet FM 的行为诊断.

生成 3 张图:
  05_tetrahedron:   4-class case (3D 四面体) 作为"还能画的最后一张"
  06_sample_shape:  Dirichlet(1,...,1) 在 K=3/10/40/129 下的采样形状 (sorted bar)
  07_signal_decay:  高 K 下 FM 训练信号退化 (vertex 距中心 vs 距离, 各维度 SNR)

都写到 docs/images/
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Serif CJK JP', 'DejaVu Sans']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
OUT_DIR = Path(__file__).resolve().parent


# =================================================================== #
# Figure 5: 4-class tetrahedron (3D simplex)
# =================================================================== #

def fig_tetrahedron():
    # 4 个 vertex (规则四面体)
    V = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=float) / np.sqrt(3)
    names = ['bg', 'part_A', 'part_B', 'part_C']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']

    def to_3d(x):  # [n, 4] simplex -> [n, 3] cartesian
        return x @ V

    fig = plt.figure(figsize=(16, 4.2))
    ts = [0.0, 0.3, 0.7, 1.0]

    for i, t in enumerate(ts):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')

        # 画四面体骨架
        for a in range(4):
            for b in range(a + 1, 4):
                ax.plot(*zip(V[a], V[b]), color='#888', lw=0.7)
        for vi, (v, n, c) in enumerate(zip(V, names, colors)):
            ax.scatter(*v, s=90, color=c, zorder=5)
            ax.text(*(v * 1.18), n, fontsize=9, ha='center')

        # 4 个真实 voxel，每个走向不同 vertex
        for target, c in enumerate(colors):
            alpha_vec = np.ones(4)
            alpha_vec[target] += t / max(1e-4, 1 - t)
            samples = np.random.dirichlet(alpha_vec, size=40)
            pts = to_3d(samples)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=14, alpha=0.5,
                       color=c, edgecolors='none')

        ax.set_title(f't = {t:.2f}', fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect([1, 1, 1])
        # 去掉多余 panel 颜色
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)

    fig.suptitle(
        "Dirichlet FM: 4-class case = 四面体 (tetrahedron, Delta^3)\n"
        "K=4 已经是能可视化的极限；K=129 的 simplex 是 128 维, 画不出来",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / 'dirichlet_fm_05_tetrahedron.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


# =================================================================== #
# Figure 6: Dirichlet(1,...,1) 采样在 K 增大时的形状
# =================================================================== #

def fig_sample_shape():
    K_values = [3, 10, 40, 129]
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))

    for ax, K in zip(axes, K_values):
        # 画 3 个独立 sample 的 sorted bar
        for i, color in enumerate(['#1f77b4', '#ff7f0e', '#2ca02c']):
            x = np.random.dirichlet(np.ones(K))
            x_sorted = np.sort(x)[::-1]
            ax.bar(np.arange(K) + i * 0.25, x_sorted, width=0.25,
                   color=color, alpha=0.75,
                   label=f'sample #{i+1}' if K == K_values[0] else None)
        # 均匀分布的理论 mean: 1/K
        ax.axhline(1.0 / K, color='red', ls='--', lw=1.2,
                   label=f'1/K = {1/K:.4f}' if K == K_values[0] else f'1/K = {1/K:.4f}')

        # 画出 one-hot vertex (对比参考)
        ax.axhline(1.0, color='#8e44ad', ls=':', lw=1.0, alpha=0.6)

        ax.set_title(f'Dirichlet(1,...,1) on Delta^{K-1}  (K = {K})', fontsize=11)
        ax.set_xlabel('class index (sorted desc)')
        if K == 3:
            ax.set_ylabel('probability')
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        if K == K_values[-1]:
            ax.legend(loc='upper right', fontsize=8)
            ax.text(
                K * 0.6, 0.6,
                '观察:\nK 增大 -> mean 下降 (1/K)\n每个 sample 仍有 1-2 个"大"分量\n其余压到近 0',
                fontsize=9, color='#333',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3cd', edgecolor='#daa520'),
            )

    fig.suptitle(
        "Dirichlet(1,1,...,1) 采样: K 增大时, 大部分维度压到 0, 只有少数'突出' "
        "(紫线=one-hot vertex 目标, 红线=均匀中心 1/K)",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    out = OUT_DIR / 'dirichlet_fm_06_sample_shape.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


# =================================================================== #
# Figure 7: 高 K 下 FM 训练信号的退化
# =================================================================== #

def fig_signal_decay():
    K_range = np.array([2, 3, 5, 10, 20, 40, 80, 129, 200, 400])

    # 1. 起点 x_0 = 均匀中心 (1/K,...,1/K) 到 vertex (0,...,1,...,0) 的 L2 距离
    l2_to_vertex = np.sqrt((1 - 1/K_range)**2 + (K_range - 1) * (1/K_range)**2)

    # 2. Target vertex 的 "活跃维度比例" = 1/K
    active_frac = 1.0 / K_range

    # 3. Per-dimension 平均速度大小 (|v_i|) 相对于维度数
    #    v = x_1 - x_0, v_target = 1 - 1/K, v_others = -1/K
    #    average |v| = (|v_target| + (K-1)*|v_others|) / K
    avg_v = (np.abs(1 - 1/K_range) + (K_range - 1) * (1/K_range)) / K_range
    # SNR ratio: v_target_magnitude / v_background_magnitude
    snr = (1 - 1/K_range) / (1/K_range)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    axes[0].plot(K_range, l2_to_vertex, 'o-', color='#1f77b4', lw=1.5)
    axes[0].axhline(1.0, color='red', ls='--', lw=1, alpha=0.6, label='极限 = 1 (K -> inf)')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('K (类别数)')
    axes[0].set_ylabel('||x_0 - x_1||_2')
    axes[0].set_title('起点到 vertex 的距离随 K 变化\n(flow 要"走"的距离)', fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    for K in [3, 40, 129]:
        idx = np.where(K_range == K)[0][0]
        axes[0].annotate(f'K={K}', (K, l2_to_vertex[idx]),
                        textcoords='offset points', xytext=(-5, -15),
                        fontsize=9, color='#333')

    axes[1].plot(K_range, active_frac * 100, 'o-', color='#2ca02c', lw=1.5)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('活跃维度比例 (%) = 100/K')
    axes[1].set_title('每个 voxel 的"正向目标维度"占比\n(one-hot 里只有 1 维是 1, 其余全是 0)', fontsize=10)
    axes[1].grid(alpha=0.3, which='both')
    for K in [3, 40, 129]:
        idx = np.where(K_range == K)[0][0]
        axes[1].annotate(f'K={K}\n{active_frac[idx]*100:.2f}%',
                        (K, active_frac[idx] * 100),
                        textcoords='offset points', xytext=(5, 5),
                        fontsize=9, color='#333')

    axes[2].plot(K_range, snr, 'o-', color='#d62728', lw=1.5)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('K')
    axes[2].set_ylabel('SNR = |v_target| / |v_bg|')
    axes[2].set_title('每个维度的"信噪比"\n(目标维的速度幅值 vs 背景维)', fontsize=10)
    axes[2].grid(alpha=0.3, which='both')
    for K in [3, 40, 129]:
        idx = np.where(K_range == K)[0][0]
        axes[2].annotate(f'K={K}\n{snr[idx]:.0f}x',
                        (K, snr[idx]),
                        textcoords='offset points', xytext=(-5, 10),
                        fontsize=9, color='#333')

    fig.suptitle(
        "高 K 对 FM 训练的影响: 距离饱和到 1, 活跃维度比例 1/K 线性下降, SNR 线性上升",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    out = OUT_DIR / 'dirichlet_fm_07_signal_decay.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out}')


if __name__ == '__main__':
    print(f'Writing figures to: {OUT_DIR}')
    fig_tetrahedron()
    fig_sample_shape()
    fig_signal_decay()
    print('done.')
