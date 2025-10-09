import matplotlib.pyplot as plt
import numpy as np

# 设置字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['lines.markersize'] = 8

# 数据
max_Latency = np.array([0.0, 8.0, 16.0, 32.0, 64.0])
avg_Latency = np.array([0.0, 3.0, 6.3, 13.2, 26.5])
std_Latency = np.array([0.0, 2.2, 3.9, 7.1, 12.6])

# GEPO_eval_acc = np.array([0.785, 0.758, 0.753, 0.745, 0.752])
# GSPO_eval_acc = np.array([0.781, 0., 0., 0., 0.718])
# GRPO_eval_acc = np.array([0.771, 0., 0., 0., 0.702])
# drGRPO_eval_acc = np.array([0.772, 0., 0., 0., 0.665])
# BNPO_eval_acc = np.array([0.741, 0., 0., 0., 0.])


GEPO_eval_acc = np.array([0.785, 0.758, 0.753, 0.745, 0.752])
GSPO_eval_acc = np.array([0.781, 0.733, 0.712, 0.717, 0.718])
GRPO_eval_acc = np.array([0.771, 0.741, 0.729, 0.713, 0.702])
drGRPO_eval_acc = np.array([0.772, 0.723, 0.711, 0.682, 0.665])
BNPO_eval_acc = np.array([0.741, 0.71, 0.712, 0.701, 0.689])

# 方法名称和颜色
method_names = ['GEPO', 'GSPO', 'GRPO', 'Dr. GRPO', 'BNPO']
colors = ["#c83e4b", "#5894c8", "#b66e1a", "#08297b", "#73bd6b"]

# 原始x位置（0,1,2,3,4）
x_pos = np.arange(len(max_Latency))  # [0,1,2,3,4]
n_latency = len(x_pos)
n_methods = len(method_names)

# === 关键：计算每组柱子的中心位置，用于对齐两个图 ===
bar_width = 0.15
x_align = x_pos + bar_width * (n_methods - 1) / 2  # 每组柱子的中心位置

# 创建子图
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 8),
    gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.15}
)

# === 图1：分组柱状图（评估精度）===
for i, (acc, color, label) in enumerate(zip(
    [GEPO_eval_acc, GSPO_eval_acc, GRPO_eval_acc, drGRPO_eval_acc, BNPO_eval_acc],
    colors, method_names)):
    ax1.bar(x_pos + i * bar_width, acc, width=bar_width, color=color, label=label, edgecolor='none')

# 设置图1
ax1.set_ylabel('Evaluation Accuracy', fontsize=14)
ax1.set_ylim(0.64, 0.80)
ax1.grid(True, alpha=0.3, axis='y')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 使用对齐的x位置设置x轴
ax1.set_xticks(x_align)
ax1.set_xticklabels([f'{int(lat)}' for lat in max_Latency])

# 添加数值标签
for i in range(n_latency):
    for j, acc in enumerate([
        GEPO_eval_acc, GSPO_eval_acc, GRPO_eval_acc, drGRPO_eval_acc, BNPO_eval_acc
    ]):
        ax1.text(x_pos[i] + j * bar_width, acc[i] + 0.005, f'{acc[i]:.3f}',
                 ha='center', va='bottom', fontsize=10, color=colors[j], weight='normal')

# 图例放在顶部
handles1, labels1 = ax1.get_legend_handles_labels()
fig.legend(
    handles1, labels1,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.96),
    ncol=5,
    frameon=False,
    fontsize=13
)

# === 图2：延迟分布（与图1 x轴对齐）===
# 使用相同的 x_align 位置绘制延迟数据（与柱状图组中心对齐）
ax2.errorbar(x_align, avg_Latency, yerr=std_Latency, fmt='s', color='tab:orange',
             linestyle='--', capsize=5, label='Avg ($\pm$Std) Latency', elinewidth=2, markerfacecolor='white')
ax2.plot(x_align, max_Latency, marker='^', linestyle=':', color='tab:red', label='Max Latency (Ref)')

ax2.set_ylabel('Latency\n(Steps)', fontsize=14)
ax2.set_ylim(-5, 70)
ax2.grid(True, alpha=0.3, axis='y')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 使用与图1完全相同的x轴刻度和标签
ax2.set_xticks(x_align)
ax2.set_xticklabels([f'{int(lat)}' for lat in max_Latency])
ax2.set_xlabel('Maximum Tolerable Latency (Steps)', fontsize=14)

# 图2图例
ax2.legend(loc='center', bbox_to_anchor=(0.5, 0.90), ncol=2, frameon=False, fontsize=12)

# === 布局调整 ===
plt.subplots_adjust(
    top=0.90,
    bottom=0.1,
    left=0.10,
    right=0.95,
    hspace=0.15
)

# ✅ 可选：设置相同的xlim避免边缘裁剪
ax1.set_xlim(x_align[0] - 0.5, x_align[-1] + 0.5)
ax2.set_xlim(x_align[0] - 0.5, x_align[-1] + 0.5)

# 保存图像
# output_pdf = 'EvalAccDelay_Aligned.pdf'
output_png = 'EvalAccDelay_Aligned.png'
# plt.savefig(output_pdf, dpi=300, bbox_inches='tight', format='pdf')
plt.savefig(output_png, dpi=300, bbox_inches='tight', format='png')

# print(f"图表已保存为 {output_pdf} 和 {output_png}")

# 显示
plt.show()