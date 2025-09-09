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


def smooth(data, window=10):
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


def smooth_ema(data, alpha=0.1):
    return pd.Series(data).ewm(alpha=alpha).mean().values
#
# def plot_online(df, savepath,
#                 mothods=["GSPO","GRPO","GEPO",],
#                 colors=["#5894c8","#f06e22","#c83e4b",]):
#     global_steps = df["train/global_step"].tolist()
#     plt.figure(figsize=[6,4])
#     for mtd,cor in zip(mothods,colors):
#         col_name = f"[paper]{mtd}_online - train/reward"
#         data_list = df[col_name].tolist()
#         n=len(data_list)
#         plt.plot(global_steps[:n], data_list,
#                  color=cor,
#                  linestyle="-",
#                  linewidth=1,
#                  alpha=0.2,
#                  label=None)
#
#         # 平滑后的数据
#         # data_smooth = smooth(data_list, window=40)
#         data_smooth = smooth_ema(data_list,alpha=0.05)
#         plt.plot(global_steps[:n], data_smooth,
#                  color=cor,
#                  linestyle="-",
#                  linewidth=3 if mtd == "GEPO" else 2,
#                  # marker="s",
#                  # markersize=1 if mtd == "GEPO" else 1,
#                  label=mtd if mtd != "GEPO" else f"{mtd} (ours)")
#
#     plt.legend(framealpha=0.4,
#                # loc='lower right',
#                fontsize=11)
#     # plt.ylim([0.4,0.8])
#     # plt.title(f"0s Latency (Online RL)")
#     plt.tight_layout()
#     plt.savefig(savepath)
#     plt.show()
#     plt.close()

def plot_online_reward(df, savepath,
                       methods=["GSPO", "GRPO", "GEPO"],
                       colors=["#5894c8", "#b66e1a", "#c83e4b"],
                       xlabel="Training Steps",
                       ylabel="KL Divergence",
                       title="KL Divergence during Online Training",
                       grid_alpha=0.2,
                       dpi=300):
    """
    绘制 train/reward 曲线，无 xlabel/ylabel/title，tick 和 legend 文字加粗，
    图例边框为灰色细线
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
        "lines.linewidth": 2,
        "figure.figsize": [7, 5],
        # "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.shadow": False,
        "legend.edgecolor": "gray",        # 图例边框颜色：灰色
        "legend.framealpha": 0.9,
        "grid.alpha": grid_alpha,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    # 确保坐标轴边框可见
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

    # 方法到标签的映射
    mth2label = {
        "GSPO": "GSPO",
        "GRPO": "GRPO",
        "GEPO": "GEPO (ours)",
    }

    for method, color in zip(methods, colors):
        col_name = f"[paper]{method}_online - train/reward"
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found in DataFrame.")
            continue

        data_list = df[col_name].dropna().tolist()
        steps = global_steps[:len(data_list)]

        # 原始数据：浅色背景
        ax.plot(steps, data_list,
                color=color,
                alpha=0.2,
                linewidth=1,
                zorder=1)

        # EMA 平滑数据
        data_smooth = smooth_ema(data_list, alpha=0.05)
        lw = 3.5 if method == "GEPO" else 2.0
        ax.plot(steps, data_smooth,
                color=color,
                linewidth=lw,
                label=mth2label[method],
                zorder=2)

    ax.set_xlabel(xlabel, fontweight='bold', fontsize=15,)
    # ax.set_ylabel(ylabel, fontweight='bold')
    # ax.set_title(title, fontweight='bold', pad=15)

    # 启用网格
    ax.grid(True, which='major', axis='y', zorder=0, alpha=grid_alpha)

    # 🔼 加粗 tick labels
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # 图例：文字加粗 + 灰色边框 + 细边框
    legend = ax.legend(
        prop={'weight': 'bold'},           # 文字加粗
        loc='lower left',
        fancybox=False,
        framealpha=0.9
    )
    legend.get_frame().set_linewidth(0.8)  # 边框细线
    # edgecolor 已在 rcParams 中设为 'gray'

    # 布局优化
    plt.tight_layout()

    # 保存高清图
    plt.savefig(savepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()



if __name__ == '__main__':
    df_online = pd.read_csv("D:\Research_HUB\GPG\open-r1\src\open_r1\GEPO_paper_plot\wandb-res\wandb_TrainReward.csv")

    # df_online =df_online[::2]
    path_online="D:\Research_HUB\GPG\open-r1\src\open_r1\GEPO_paper_plot\wandb-res\TrainReward.pdf"


    plot_online_reward(df_online, path_online)
