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

# 或者：指数移动平均（EMA，更平滑）
def smooth_ema(data, alpha=0.01):
    return pd.Series(data).ewm(alpha=alpha).mean().values

def smooth(data, window=10):
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values
# def plot_online(df, savepath,
#                 mothods=["GEPO_diff4","GEPO_diff32",],
#                 colors=["#5894c8","#c83e4b",]):
#     global_steps = df["train/global_step"].tolist()
#     plt.figure(figsize=[6,4])
#     for mtd,cor in zip(mothods,colors):
#         col_name = f"[paper]{mtd} - train/step_diff"
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
#         # data_smooth = smooth(data_list, window=60)
#         data_smooth = smooth_ema(data_list, alpha=0.05)
#
#         mth2label = {"GEPO_diff4": "GEPO-MaxDelay 8",
#                      "GEPO_diff32": "GEPO-MaxDelay 64",
#                      }
#
#         plt.plot(global_steps[:n], data_smooth,
#                  color=cor,
#                  linestyle="-",
#                  linewidth=3,
#                  # marker="s",
#                  # markersize=1 if mtd == "GEPO" else 1,
#                  label=mth2label[mtd] ,
#                  )
#
#     plt.legend(framealpha=0.4, loc='upper left', fontsize=11,)
#     # plt.ylim([0.4,0.8])
#     # plt.title(f"0s Latency (Online RL)")
#     plt.xticks(fontweight='bold')
#     plt.yticks(fontweight='bold')
#     plt.tight_layout()
#     plt.savefig(savepath)
#     plt.show()
#     plt.close()


def plot_online(df, savepath,
                methods=["GEPO_diff4", "GEPO_diff32"],
                colors=["#5894c8", "#c83e4b"],
                xlabel="Training Steps",
                ylabel="Step Difference",
                title="Online RL Training Dynamics",
                grid_alpha=0.3,
                dpi=300):
    """
    绘制在线训练 step_diff 曲线，带 EMA 平滑
    """
    global_steps = df["train/global_step"].tolist()

    # 设置全局字体和分辨率
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.2,
        "xtick.direction": "in",  # 刻度线向内
        "ytick.direction": "in",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "lines.linewidth": 2,
        "figure.figsize": [7, 5],
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.shadow": False,
        "legend.edgecolor": "black",
        "legend.framealpha": 0.8,
        "grid.alpha": grid_alpha,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    # 方法到标签的映射
    mth2label = {
        "GEPO_diff4": "GEPO-MaxDelay 8",
        "GEPO_diff32": "GEPO-MaxDelay 64",
    }

    for method, color in zip(methods, colors):
        col_name = f"[paper]{method} - train/step_diff"
        if col_name not in df.columns:
            print(f"Warning: {col_name} not found in DataFrame.")
            continue

        data_list = df[col_name].dropna().tolist()  # 避免 NaN
        steps = global_steps[:len(data_list)]

        # 原始数据：浅色细线
        ax.plot(steps, data_list,
                color=color,
                alpha=0.2,
                linewidth=1,
                zorder=1)

        # EMA 平滑数据：主线条
        data_smooth = smooth_ema(data_list, alpha=0.05)
        ax.plot(steps, data_smooth,
                color=color,
                linewidth=2.5,
                label=mth2label.get(method, method),
                zorder=2)

    # 添加网格
    ax.grid(True, which='major', axis='y', zorder=0)
    # 🔼 加粗 x 和 y 的 tick labels
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    # 设置坐标轴
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=15,)
    # ax.set_ylabel(ylabel, fontweight='bold')
    # ax.set_title(title, fontweight='bold', pad=15)

    # 图例：文字加粗 + 灰色边框 + 细边框
    legend = ax.legend(
        prop={'weight': 'bold'},           # 文字加粗
        loc='upper left',
        fancybox=False,
        edgecolor='gray',
        framealpha=0.9
    )
    legend.get_frame().set_linewidth(0.8)  # 边框细线
    # 图例
    # ax.legend(loc='lower right', fontsize=11, edgecolor='gray', fancybox=False)
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    # 布局优化
    plt.tight_layout()

    # 保存高清图
    plt.savefig(savepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()


if __name__ == '__main__':
    df_online = pd.read_csv("D:\Research_HUB\GPG\open-r1\src\open_r1\GEPO_paper_plot\wandb-res\wandb_DelayStep.csv")

    # df_online =df_online[::2]
    path_online="D:\Research_HUB\GPG\open-r1\src\open_r1\GEPO_paper_plot\wandb-res\DelayStep.pdf"


    plot_online(df_online, path_online)
