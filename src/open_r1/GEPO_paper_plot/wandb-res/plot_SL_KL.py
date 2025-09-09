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
# 或者：指数移动平均（EMA，更平滑）
def smooth_ema(data, alpha=0.01):
    return pd.Series(data).ewm(alpha=alpha).mean().values

def plot_online_kl(df, savepath,
                   methods=["GEPO_diff4", "GEPO_diff32"],
                   colors=["#5894c8", "#c83e4b"],
                   xlabel="Training Steps",
                   ylabel="KL Divergence",
                   title="KL Divergence during Online Training",
                   grid_alpha=0.2,
                   dpi=300):
    """
    绘制 train/cppo_kl 曲线，带 EMA 平滑，对数Y轴，tick和legend文字加粗，图例边框细
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
        "axes.linewidth": 1.2,              # 坐标轴边框线宽
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
        "legend.edgecolor": "black",
        "legend.framealpha": 0.9,
        "grid.alpha": grid_alpha,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
    })

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)

    # 确保边框可见
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

    # 方法到标签的映射
    mth2label = {
        "GEPO_diff4": "GEPO-MaxDelay 8",
        "GEPO_diff32": "GEPO-MaxDelay 64",
    }

    for method, color in zip(methods, colors):
        col_name = f"[paper]{method} - train/cppo_kl"
        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found in DataFrame.")
            continue

        data_list = df[col_name].dropna().tolist()
        steps = global_steps[:len(data_list)]

        # 原始数据：浅色细线
        ax.plot(steps, data_list,
                color=color,
                alpha=0.2,
                linewidth=1,
                zorder=1)

        # EMA 平滑数据
        data_smooth = smooth_ema(data_list, alpha=0.05)
        ax.plot(steps, data_smooth,
                color=color,
                linewidth=3,
                label=mth2label[method],
                zorder=2)

    # 设置对数 Y 轴
    ax.set_yscale('log')

    # 设置坐标轴标签
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=15,)
    # ax.set_ylabel(ylabel, fontweight='bold')
    # ax.set_title(title, fontweight='bold', pad=15)

    # 启用网格（对数坐标下也有效）
    ax.grid(True, which='major', axis='y', alpha=grid_alpha, linestyle='-', linewidth=0.8)
    ax.grid(True, which='minor', axis='y', alpha=grid_alpha * 0.6, linestyle='--', linewidth=0.4)

    # 加粗 tick labels
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # 图例：文字加粗，边框细
    legend = ax.legend(
        prop={'weight': 'bold'},           # 文字加粗
        loc='upper left',                  # log下KL下降，放左下更合适
        edgecolor='gray',
        fancybox=False,
        framealpha=0.9
    )
    legend.get_frame().set_linewidth(0.8)  # 边框细线

    # 布局优化
    plt.tight_layout()

    # 保存高清图
    plt.savefig(savepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close()



if __name__ == '__main__':
    df_online = pd.read_csv("D:\Research_HUB\GPG\open-r1\src\open_r1\GEPO_paper_plot\wandb-res\wandb_LearnerSamplerKL.csv")

    df_online =df_online[::2]
    path_online="D:\Research_HUB\GPG\open-r1\src\open_r1\GEPO_paper_plot\wandb-res\LearnerSamplerKL.pdf"


    plot_online_kl(df_online, path_online)
