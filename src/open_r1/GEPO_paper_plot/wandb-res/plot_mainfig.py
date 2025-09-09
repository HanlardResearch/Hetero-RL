import matplotlib.pyplot as plt
import pandas as pd

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



def plot_online(df, savepath,
                mothods=["GSPO","GRPO","BNPO","drGRPO","GEPO",],
                colors=["#5894c8","#b66e1a","#73bd6b","#08297b","#c83e4b",]):
    global_steps = df["train/global_step"].tolist()
    plt.figure(figsize=[6,4])
    for mtd,cor in zip(mothods,colors):
        col_name = f"[paper]{mtd}_online - eval/rewards/accuracy_reward_lv35/mean"
        data_list = df[col_name].tolist()
        n=len(data_list)
        plt.plot(global_steps[:n], data_list,
                 color=cor,
                 linestyle="-",
                 linewidth=2.5 if mtd=="GEPO" else 2,
                 marker="s",
                 markeredgewidth=0.8,
                 markeredgecolor="white",
                 markersize=6 if mtd == "GEPO" else 4,
                 label=mtd if mtd != "GEPO" else f"{mtd}")
    plt.legend(framealpha=0.4,
               # loc='lower right',
               bbox_to_anchor=(0.145, 0.6),
               fontsize=11)
    plt.ylim([0.4,0.8])
    plt.title(f"0s Latency (Online RL)", fontweight='bold')
    plt.xlabel("Training steps", fontweight='bold')
    plt.ylabel("Eval Accuracy", fontweight='bold')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()
    plt.close()

def plot_diff32(df, savepath,
                mothods=["GSPO", "GRPO", "GEPO", ],
                colors=["#5894c8", "#b66e1a",  "#c83e4b", ]):

    global_steps = df["train/global_step"].tolist()
    plt.figure(figsize=[6, 4])
    for mtd,cor in zip(mothods,colors):
        col_name = f"[paper]{mtd}_diff32 - eval/rewards/accuracy_reward_lv35/mean"
        data_list = df[col_name].tolist()
        n=len(data_list)
        plt.plot(global_steps[:n], data_list,
                 color=cor,
                 linestyle="-",
                 linewidth=2.5 if mtd=="GEPO" else 2,
                 marker="s",
                 markeredgewidth=0.8,
                 markeredgecolor="white",
                 markersize=6 if mtd == "GEPO" else 4,
                 label=mtd if mtd != "GEPO" else f"{mtd}")
    plt.legend(framealpha=0.4,
               # loc='center left',
               bbox_to_anchor=(0.19, 0.5),
               fontsize=11)
    plt.ylim([0.4, 0.8])
    plt.title(f"Max-1800s Latency (Hetero RL)", fontweight='bold')
    plt.xlabel("Training steps", fontweight='bold')
    plt.ylabel("Eval Accuracy", fontweight='bold')
    plt.tight_layout()
    plt.savefig(savepath)

    plt.show()
    plt.close()


if __name__ == '__main__':
    df_diff32 = pd.read_csv("D:\Research_HUB\GPG\open-r1\src\open_r1\GEPO_paper_plot\wandb-res\wandb_diff32.csv")
    df_online = pd.read_csv("D:\Research_HUB\GPG\open-r1\src\open_r1\GEPO_paper_plot\wandb-res\wandb_online.csv")

    df_online =df_online[::2]
    # df_diff32 = df_diff32[::2]
    path_diff32="D:\Research_HUB\GPG\open-r1\src\open_r1\GEPO_paper_plot\wandb-res\wandb_diff32.svg"
    path_online="D:\Research_HUB\GPG\open-r1\src\open_r1\GEPO_paper_plot\wandb-res\wandb_online.svg"


    plot_online(df_online, path_online)
    plot_diff32(df_diff32, path_diff32)