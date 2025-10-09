import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D  # 用于创建图例句柄
# 5. 创建完美匹配的图例
from matplotlib.patches import Rectangle

def variance_difference(a, b):
    """
    返回标准方法方差 减去 新方法方差
    """
    old = variance_standard(a, b)
    new = variance_new(a, b)
    return old - new


def kl_divergence(a, b):
    """计算KL散度"""
    epsilon = 1e-10
    a = np.clip(a, epsilon, 1 - epsilon)
    b = np.clip(b, epsilon, 1 - epsilon)
    kl = a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    return kl


def variance_standard(a, b):
    """
    计算标准重要性采样权重的方差
    权重: w(1) = a / b, w(2) = (1 - a) / (1 - b)
    Var = E[w^2] - (E[w])^2 = E[w^2] - 1
    """

    term1 = (a ** 2) / b
    term2 = ((1 - a) ** 2) / (1 - b)
    E_w2 = term1 + term2
    return E_w2 - 1.0


def variance_new2(a, b):
    """
    计算你提出的新重要性采样权重的方差
    权重: w(x) = p(x) / E_q[q(x)], 其中 E_q[q(x)] = b^2 + (1 - b)^2
    """
    c = b ** 2 + (1 - b) ** 2  # E_q[q(x)]
    # 权重取值
    w1 = a / c
    w2 = (1 - a) / c

    # E[w] under q
    E_w = b * w1 + (1 - b) * w2

    # E[w^2] under q
    E_w2 = b * (w1 ** 2) + (1 - b) * (w2 ** 2)

    return E_w2 - E_w ** 2


def variance_new(a, b, epsilon=1e-8):
    """
    计算归一化的新重要性采样权重在伯努利分布下的方差
    权重定义为: w(x) = p(x) / E_q[p], 其中 E_q[p] = a*b + (1-a)*(1-b)
    该权重是无偏的，且方差有解析解。

    参数:
        a (float or array): 目标分布参数 p(x) ~ Bernoulli(a), a ∈ (0,1)
        b (float or array): 提议分布参数 q(x) ~ Bernoulli(b), b ∈ (0,1)
        epsilon (float): 防止除零的小量

    返回:
        var (float or array): 方差 Var_q(w)
    """
    # 将输入转换为 numpy 数组以支持向量化操作
    a = np.asarray(a)
    b = np.asarray(b)

    # 防止 a 或 b 超出 (0,1)
    a = np.clip(a, epsilon, 1 - epsilon)
    b = np.clip(b, epsilon, 1 - epsilon)

    numerator = a + b - 2 * a * b
    denominator = 1 - a - b + 2 * a * b

    # 防止分母过小
    denominator = np.maximum(denominator, epsilon)

    var = numerator / denominator
    return var

# 创建网格
a_range = np.linspace(0.05, 0.95, 50)
b_range = np.linspace(0.05, 0.95, 50)
A, B = np.meshgrid(a_range, b_range)

Z = variance_difference(A, B)  # 方差差值
old_var = variance_standard(A, B)
new_var = variance_new(A, B)
new_var2 = variance_new2(A, B)
KL = kl_divergence(A, B)

# 创建图形
fig = plt.figure(figsize=(16, 4))
gs = GridSpec(1, 4, figure=fig,
              width_ratios=[2.4, 2.4, 2.4, 1.8],
              wspace=0.4,
              left=0.08, right=0.94,
              bottom=0.15, top=0.9)


###################################### 图 1 ######################################

ax4 = fig.add_subplot(gs[0], projection='3d')

# 创建自定义colormap确保最佳对比度
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# 标准方法: 从浅黄到深红 (表示方差从低到高)
colors_old = [(1, 1, 0.7), (1, 0.8, 0.5), (1, 0.5, 0.2), (0.8, 0, 0)]  # 黄->橙->红
cmap_old = LinearSegmentedColormap.from_list("standard_cmap", colors_old, N=256)

# 新方法: 从浅蓝到深蓝 (表示方差从低到高，但整体保持低位)
colors_new = [(0.9, 0.95, 1), (0.7, 0.85, 1), (0.4, 0.6, 0.9), (0.1, 0.3, 0.7)]  # 浅蓝->深蓝
cmap_new = LinearSegmentedColormap.from_list("new_cmap", colors_new, N=256)

# 2. 绘制曲面 (优化视觉层次)
surface_old = ax4.plot_surface(
    A, B, old_var,
    cmap=cmap_old,
    alpha=0.65,
    rstride=2, cstride=2,
    linewidth=0.6,
    edgecolor='darkred'
)

surface_new = ax4.plot_surface(
    A, B, new_var,
    cmap=cmap_new,
    alpha=0.4,
    rstride=2, cstride=2,
    linewidth=0.6,
    edgecolor='#E040FB'
)

# === 新增：标记 new_var > old_var 的点 ===
# 找出 new_var > old_var 的位置
mask = new_var > old_var

# 将这些点展平，避免空数组报错
A_flat = A[mask]
B_flat = B[mask]
new_var_flat = new_var[mask]

# 如果有满足条件的点，则绘制为红色/橙色散点
if len(A_flat) > 0:
    ax4.scatter(A_flat, B_flat, new_var_flat,
                color='#00EB76',  # 或 'red'
                alpha=0.3,
                s=1,           # 点大小\mathbb{E}_q p
                edgecolors='#00EB76',
                linewidth=0.5,
                label=r'$\mathrm{Var}(\frac{p}{\mathbb{E}_q p}) > \mathrm{Var}(\frac{p}{q})$')

# === 可选：在底部投影一个等高线图来强调这些区域 ===
# ax4.contour(A, B, (new_var - old_var), levels=[0], colors='orange', linestyles='--', offset=Z_min)

# 图例
legend_elements = [
    Rectangle((0, 0), 1, 1,
              fc='darkred',
              edgecolor='darkred',
              linewidth=1.0,
              alpha=0.7,
              label=r'Standard IS: $\mathrm{Var}(\frac{p}{q})$'),

    Rectangle((0, 0), 1, 1,
              fc='#E040FB',
              edgecolor='#E040FB',
              linewidth=1.0,
              alpha=0.5,
              label=r'Unbias: $\mathrm{Var}(\frac{p}{\mathbb{E}_q p})$'),
]

# 如果上面加了 scatter，也要加入图例
if 'ax4.scatter' in locals() or len(A_flat) > 0:
    legend_elements.append(
        Rectangle((0, 0), 1, 1,
                  fc='#00EB76',
                  edgecolor='#00EB76',
                  linewidth=0.5,
                  alpha=0.6,
                  label=r'$\mathrm{Var}(\frac{p}{\mathbb{E}_q p}) > \mathrm{Var}(\frac{p}{q})$')
    )

ax4.legend(handles=legend_elements,
           loc='upper right',
           fontsize=10,
           frameon=True,
           framealpha=0.95,
           facecolor='white',
           edgecolor='gray'
           )

# 添加 a=b 参考线
a_equal_b = np.linspace(0.05, 0.95, 50)
z_old_equal_b = [variance_standard(a_val, a_val) for a_val in a_equal_b]
ax4.plot(a_equal_b, a_equal_b, z_old_equal_b, 'r--', linewidth=3.0, alpha=1.0)

# 设置坐标轴
ax4.set_xlabel('a', labelpad=12, fontsize=12)
ax4.set_ylabel('b', labelpad=12, fontsize=12)
ax4.set_zlabel('Variance', labelpad=6, fontsize=14)
ax4.set_title('Variance Comparison', fontsize=14, pad=20)
ax4.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)

###################################### 图 1 ######################################

###################################### 图 2 ######################################
# --- 2. KL 散度曲面图 ---
ax2 = fig.add_subplot(gs[1], projection='3d')

# 创建自定义colormap确保最佳对比度
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# 标准方法: 从浅黄到深红 (表示方差从低到高)
colors_old = [(1, 1, 0.7), (1, 0.8, 0.5), (1, 0.5, 0.2), (0.8, 0, 0)]  # 黄->橙->红
cmap_old = LinearSegmentedColormap.from_list("standard_cmap", colors_old, N=256)

# 新方法: 从浅蓝到深蓝 (表示方差从低到高，但整体保持低位)
colors_new = [(0.9, 0.95, 1), (0.7, 0.85, 1), (0.4, 0.6, 0.9), (0.1, 0.3, 0.7)]  # 浅蓝->深蓝
cmap_new = LinearSegmentedColormap.from_list("new_cmap", colors_new, N=256)

# 2. 绘制曲面 (优化视觉层次)
surface_old = ax2.plot_surface(
    A, B, new_var,
    cmap=cmap_old,
    alpha=0.65,
    rstride=2, cstride=2,
    linewidth=0.6,
    edgecolor='#E040FB'
)

surface_new = ax2.plot_surface(
    A, B, new_var2,# ours
    cmap=cmap_new,
    alpha=0.4,
    rstride=2, cstride=2,
    linewidth=0.6,
    edgecolor='darkblue'
)

# === 新增：标记 new_var > old_var 的点 ===
# 找出 new_var > old_var 的位置
mask = new_var2 > new_var

# 将这些点展平，避免空数组报错
A_flat = A[mask]
B_flat = B[mask]
new_var_flat = new_var[mask]

# 如果有满足条件的点，则绘制为红色/橙色散点
if len(A_flat) > 0:
    ax4.scatter(A_flat, B_flat, new_var_flat,
                color='#00EB76',  # 或 'red'
                alpha=0.3,
                s=1,           # 点大小\mathbb{E}_q p
                edgecolors='#00EB76',
                linewidth=0.5,
                label=r'$\mathrm{Var}(\frac{p}{\mathbb{E}_q p}) > \mathrm{Var}(\frac{p}{q})$')

# === 可选：在底部投影一个等高线图来强调这些区域 ===
# ax4.contour(A, B, (new_var - old_var), levels=[0], colors='orange', linestyles='--', offset=Z_min)

# 图例
legend_elements = [
    Rectangle((0, 0), 1, 1,
              fc='#E040FB',
              edgecolor='#E040FB',
              linewidth=1.0,
              alpha=0.7,
              label=r'Unbias: $\mathrm{Var}(\frac{p}{\mathbb{E}_q p})$'),

    Rectangle((0, 0), 1, 1,
              fc='darkblue',
              edgecolor='darkblue',
              linewidth=1.0,
              alpha=0.5,
              label=r'Ours: $\mathrm{Var}(\frac{p}{\mathbb{E}_q q})$'),
]

# 如果上面加了 scatter，也要加入图例
if 'ax2.scatter' in locals() or len(A_flat) > 0:
    legend_elements.append(
        Rectangle((0, 0), 1, 1,
                  fc='#00EB76',
                  edgecolor='#00EB76',
                  linewidth=0.5,
                  alpha=0.6,
                  label=r'$\mathrm{Var}(\frac{p}{\mathbb{E}_q q}) > \mathrm{Var}(\frac{p}{\mathbb{E}_q p})$')
    )

ax2.legend(handles=legend_elements,
           loc='upper right',
           fontsize=10,
           frameon=True,
           framealpha=0.95,
           facecolor='white',
           edgecolor='gray'
           )

# 添加 a=b 参考线
a_equal_b = np.linspace(0.05, 0.95, 50)
z_old_equal_b = [variance_standard(a_val, a_val) for a_val in a_equal_b]
ax2.plot(a_equal_b, a_equal_b, z_old_equal_b, 'r--', linewidth=3.0, alpha=1.0)

# 设置坐标轴
ax2.set_xlabel('a', labelpad=12, fontsize=12)
ax2.set_ylabel('b', labelpad=12, fontsize=12)
ax2.set_zlabel('Variance', labelpad=6, fontsize=14)
ax2.set_title('Variance Comparison', fontsize=14, pad=20)
ax2.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
###################################### 图 2 ######################################



###################################### 图 3 ######################################
ax1 = fig.add_subplot(gs[2], projection='3d')
surface1 = ax1.plot_surface(A, B, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('a', fontsize=12, labelpad=10)
ax1.set_ylabel('b', fontsize=12, labelpad=10)
ax1.set_zlabel(r'$\Delta$Var = Var($\frac{p}{q}$) - Var($\frac{p}{\mathbb{E}_q p}$)', fontsize=12, labelpad=6)
ax1.set_title('Variance Reduction', fontsize=14, pad=20)
ax1.view_init(elev=20, azim=30)
# fig.colorbar(surface1, ax=ax1, shrink=0.6, pad=0.05)
# a=b 线
a_equal_b = np.linspace(0.05, 0.95, 50)
z_equal_b = [variance_standard(a_val, a_val) for a_val in a_equal_b]
ax1.plot(a_equal_b, a_equal_b, z_equal_b, 'r--', linewidth=2, label='a = b')
ax1.legend(fontsize=12)
###################################### 图 3 ######################################



###################################### 图 4 ######################################
ax3 = fig.add_subplot(gs[3])
scatter = ax3.scatter(KL.flatten(), Z.flatten(), c=np.abs(A - B).flatten(),
                      cmap='coolwarm', alpha=0.6, s=1)
ax3.set_xlabel('KL Divergence', fontsize=12)
ax3.set_ylabel(r'$\Delta$Var = Var($\frac{p}{q}$) - Var($\frac{p}{\mathbb{E}_q p}$)',
               rotation=270,
               labelpad=20,  # 关键：增加此值直到不重叠（35通常足够）
               fontsize=12, )  # 额外增加标签与轴的距离

ax3.set_title(r'KL vs Variance Reduction', fontsize=14, pad=20)
ax3.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
# 限制y轴范围，避免被极端值拉伸
z_min, z_max = Z.min(), Z.max()
padding = (z_max - z_min) * 0.05
ax3.set_ylim(z_min - padding, z_max + padding)
cbar = plt.colorbar(scatter, ax=ax3, shrink=0.6, pad=0.05)
cbar.set_label('|a-b|', rotation=270, labelpad=15)
# 添加红色虚线段 y=0，x从0到2
ax3.plot([0, 2.7], [0, 0], 'r--', linewidth=1.5, label=r'$\Delta$Var = 0')
ax3.legend(fontsize=12)
###################################### 图 4 ######################################

ax4.view_init(elev=20, azim=30)  # 优化视角
ax2.view_init(elev=20, azim=30)  # 优化视角
# ax4.set_zlim(-1, 15)
# 调整布局
plt.subplots_adjust(left=0.08, right=0.94, top=0.9, bottom=0.15, wspace=0.4)
plt.savefig("./mois_figs/KL_VR_Eqp.pdf", bbox_inches='tight')
plt.show()