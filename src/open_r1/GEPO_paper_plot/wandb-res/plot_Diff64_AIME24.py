import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置全局样式（可选）
add_size = 3
plt.rcParams.update({
    "font.size": 12+add_size,
    "axes.labelsize": 12+add_size,
    "axes.titlesize": 13+add_size,
    "legend.fontsize": 11+add_size,
    "xtick.labelsize": 11+add_size,
    "ytick.labelsize": 11+add_size,
    "font.family": "serif",  # 或 "sans-serif"
    "axes.grid": True,       # 默认开启网格
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "lines.markersize": 4,
    "lines.linewidth": 1.2
})



def plot_onlineMath500(df, savepath,
                        methods=["GSPO", "GRPO", "BNPO", "dr_GRPO", "GEPO", ],
                        colors=["#5894c8", "#b66e1a", "#73bd6b", "#08297b", "#c83e4b", ],
                         xlabel="Training Steps",
                         ylabel="Accuracy Reward (Level 3.5)",
                         title="Online RL Training: Accuracy Convergence",
                         grid_alpha=0.3,
                         dpi=300):
    """
    绘制 eval/rewards/accuracy_reward_lv35/mean 曲线
    """
    global_steps = df["train/global_step"].tolist()

    # 全局样式设置
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "lines.linewidth": 2.5,
        "lines.markersize": 6,
        "lines.markeredgecolor": "white",
        "lines.markeredgewidth": 0.8,
        "figure.figsize": [7, 5],
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.shadow": False,
        "legend.edgecolor": "black",
        "legend.framealpha": 0.9,
        "grid.alpha": grid_alpha,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    # 方法到标签的映射
    # mth2label = {
    #     "GEPO_diff4": "GEPO-MaxDelay 8",
    #     "GEPO_diff32": "GEPO-MaxDelay 64",
    # }

    for method, color in zip(methods, colors):
        col_name = f"[benchmark_diff32]{method} - eval/eval_aime_2024_rewards/accuracy_reward_lv35/mean"
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found in DataFrame.")
            continue

        data_list = df[col_name].dropna().tolist()
        steps = global_steps[:len(data_list)]

        # 可选：使用 EMA 平滑（如果你希望曲线更平滑）
        # data_smooth = smooth_ema(data_list, alpha=0.05)
        # ax.plot(steps, data_smooth, color=color, linewidth=3, label=mth2label.get(method))

        # 直接使用原始数据 + marker（因为你希望看到每个点）
        ax.plot(steps, data_list,
                color=color,
                linestyle="-",
                linewidth=2.5,
                marker="s",
                markersize=6,
                markeredgewidth=0.8,
                markeredgecolor="white",
                label=method,
                zorder=2)

    # 设置坐标轴
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=15,)
    # ax.set_ylabel(ylabel, fontweight='bold')
    # ax.set_title(title, fontweight='bold', pad=15)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')

    # ax.set_ylim([0.4, 0.83])

    # 添加网格
    ax.grid(True, which='major', axis='y', zorder=0, alpha=grid_alpha)
    # 图例：文字加粗 + 灰色边框 + 细边框
    legend = ax.legend(
        prop={'weight': 'bold'},           # 文字加粗
        loc='upper left',
        # bbox_to_anchor=(0., 0.45),
        fancybox=False,
        edgecolor='gray',
        framealpha=0.9
    )
    legend.get_frame().set_linewidth(0.8)  # 边框细线
    # 图例
    # ax.legend(loc='lower right', fontsize=11, edgecolor='gray', fancybox=False)

    # 布局优化
    plt.tight_layout()

    # 保存高清图（支持 PNG/PDF）
    plt.savefig(savepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()

if __name__ == '__main__':
    df_online = pd.read_csv("D:\Research_HUB\GPG\open-r1\src\open_r1\GEPO_paper_plot\wandb-res\wandb_diff64_aime24.csv")

    # df_online =df_online[::2]
    path_online="D:\Research_HUB\GPG\open-r1\src\open_r1\GEPO_paper_plot\wandb-res\diff64_aime24.pdf"


    plot_onlineMath500(df_online, path_online)
