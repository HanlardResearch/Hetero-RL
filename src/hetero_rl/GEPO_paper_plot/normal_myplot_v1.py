import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D  # 用于创建图例句柄
# 5. 创建完美匹配的图例
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from tqdm import tqdm

N_points=41
range_min = -1.0
range_max = 1.0


def variance_difference(a, b):
    """
    返回标准方法方差 减去 新方法方差
    """
    old = variance_standard(a, b)
    new = variance_new(a, b)
    return old - new


def kl_divergence_1d(b, a):
    """
    D_KL(p || q) for p = N(b,1), q = N(a,1)
    Formula: 0.5 * (b - a)^2
    """
    return 0.5 * (b - a) ** 2


def gaussian_pdf(x, mu):
    """标准差为1的正态分布 PDF"""
    return norm.pdf(x, loc=mu, scale=1)

def variance_standard(a, b):
    """
    标准重要性采样的方差: Var = E_q[(p/q)^2] - 1
    """
    def integrand(x):
        p = gaussian_pdf(x, b)
        q = gaussian_pdf(x, a)
        if q < 1e-30:
            return 0.0
        return (p ** 2) / q

    integral, err = quad(integrand, -np.inf, np.inf, limit=100, epsabs=1e-4, epsrel=1e-4)
    return max(integral - 1.0, 0)  # 防止数值误差导致负值


def variance_new(a, b):
    """
    计算新方法在 p=N(b,1), q=N(a,1) 下的方差
    Var_new = [ E_q[p^2] - (E_q[p])^2 ] / (E_q[q])^2
    """
    # 1. 计算 E_q[q(x)] = \int q(x)^2 dx
    def q_sq(x):
        q = gaussian_pdf(x, a)
        return q * q

    E_q_q, _ = quad(q_sq, -np.inf, np.inf, limit=100)
    A = E_q_q ** 2  # (E_q[q])^2

    # 2. 计算 E_q[p(x)] = \int p(x) q(x) dx
    def p_q(x):
        p = gaussian_pdf(x, b)
        q = gaussian_pdf(x, a)
        return p * q

    E_q_p, _ = quad(p_q, -np.inf, np.inf, limit=100)
    B = E_q_p ** 2  # (E_q[p])^2

    # 3. 计算 E_q[p(x)^2] = \int p(x)^2 q(x) dx
    def p_sq_q(x):
        p = gaussian_pdf(x, b)
        q = gaussian_pdf(x, a)
        return (p ** 2) * q

    E_q_p_sq, _ = quad(p_sq_q, -np.inf, np.inf, limit=100)

    # 4. 计算方差
    numerator = E_q_p_sq - B  # Var_q(p)
    # if A < 1e-30:
    #     return np.inf
    return numerator / A

# 创建网格
# a_range = np.linspace(0.05, 0.95, 50)
# b_range = np.linspace(0.05, 0.95, 50)
# A, B = np.meshgrid(a_range, b_range)
#
# Z = variance_difference(A, B)  # 方差差值
# old_var = variance_standard(A, B)
# new_var = variance_new(A, B)
# KL = kl_divergence(A, B)

# =================== 生成网格数据 ===================

# 参数范围
a_vals = np.linspace(range_min, range_max, N_points)  # q 的均值
b_vals = np.linspace(range_min, range_max, N_points)  # p 的均值
A, B = np.meshgrid(a_vals, b_vals)
# 初始化方差矩阵
Var_std = np.zeros_like(A)
Var_new = np.zeros_like(A)
print("正在计算方差曲面（可能需要几分钟）...")
for i in tqdm(range(A.shape[0])):
    for j in range(A.shape[1]):
        a = A[i, j]
        b = B[i, j]
        Var_std[i, j] = variance_standard(a, b)
        Var_new[i, j] = variance_new(a, b)

        # 限制最大值，避免曲面失真
        Var_std[i, j] = Var_std[i, j]
        Var_new[i, j] = Var_new[i, j]
a_equal_b = np.linspace(range_min, range_max, N_points)
print("计算完成！")


# =================== 生成网格数据 ==================
Z = Var_std - Var_new
# 计算 KL 散度
KL = kl_divergence_1d(B, A)  # KL(p || q)

old_var = Var_std
new_var = Var_new

# 创建图形
fig = plt.figure(figsize=(16, 4))
gs = GridSpec(1, 4, figure=fig,
              width_ratios=[2.4, 2.4, 2.4, 1.8],
              wspace=0.4,
              left=0.08, right=0.94,
              bottom=0.15, top=0.9)

a_equal_b = np.linspace(range_min, range_max, N_points)
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
    edgecolor='darkblue'
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
                color='green',  # 或 'red'
                alpha=0.5,
                s=1.0,           # 点大小
                edgecolors='green',
                linewidth=0.5,
                label=r'$\mathrm{Var}(\frac{p}{\mathbb{E}_q q}) > \mathrm{Var}(\frac{p}{q})$')

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
              fc='darkblue',
              edgecolor='darkblue',
              linewidth=1.0,
              alpha=0.5,
              label=r'Ours: $\mathrm{Var}(\frac{p}{\mathbb{E}_q q})$'),
]

# 如果上面加了 scatter，也要加入图例
if 'ax4.scatter' in locals() or len(A_flat) > 0:
    legend_elements.append(
        Rectangle((0, 0), 1, 1,
                  fc='green',
                  edgecolor='green',
                  linewidth=0.5,
                  alpha=0.6,
                  label=r'$\mathrm{Var}(\frac{p}{\mathbb{E}_q q}) > \mathrm{Var}(\frac{p}{q})$')
    )

ax4.legend(handles=legend_elements,
           loc='upper right',
           fontsize=10,
           frameon=True,
           framealpha=0.95,
           facecolor='white',
           edgecolor='gray'
           )


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
surface2 = ax2.plot_surface(A, B, KL, cmap='plasma', alpha=0.8)
ax2.set_xlabel('a', fontsize=12, labelpad=10)
ax2.set_ylabel('b', fontsize=12, labelpad=10)
ax2.set_zlabel('KL(p||q)', fontsize=12, labelpad=10)
ax2.set_title('KL Divergence', fontsize=14, pad=20)
ax2.view_init(elev=20, azim=30)
# fig.colorbar(surface2, ax=ax2, shrink=0.6, pad=0.05)
ax2.plot(a_equal_b, a_equal_b, np.zeros_like(a_equal_b), 'r--', linewidth=2, label='a = b')
ax2.legend(fontsize=12)
###################################### 图 2 ######################################



###################################### 图 3 ######################################
ax1 = fig.add_subplot(gs[2], projection='3d')
surface1 = ax1.plot_surface(A, B, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('a', fontsize=12, labelpad=10)
ax1.set_ylabel('b', fontsize=12, labelpad=10)
ax1.set_zlabel(r'$\Delta$Var = Var($\frac{p}{q}$) - Var($\frac{p}{\mathbb{E}_q q}$)', fontsize=12, labelpad=6)
ax1.set_title('Variance Reduction', fontsize=14, pad=20)
ax1.view_init(elev=20, azim=30)
# fig.colorbar(surface1, ax=ax1, shrink=0.6, pad=0.05)
# a=b 线
z_equal_b = [variance_standard(a_val, a_val) for a_val in a_equal_b]
ax1.plot(a_equal_b, a_equal_b, z_equal_b, 'r--', linewidth=2, label='a = b')
ax1.legend(fontsize=12)
###################################### 图 3 ######################################



###################################### 图 4 ######################################
ax3 = fig.add_subplot(gs[3])
scatter = ax3.scatter(KL.flatten(), Z.flatten(), c=np.abs(A - B).flatten(),
                      cmap='coolwarm', alpha=1.0, s=5.0)
ax3.set_xlabel('KL Divergence', fontsize=12)
ax3.set_ylabel(r'$\Delta$Var = Var($\frac{p}{q}$) - Var($\frac{p}{\mathbb{E}_q q}$)',
               rotation=270,
               labelpad=20,  # 关键：增加此值直到不重叠（35通常足够）
               fontsize=12, )  # 额外增加标签与轴的距离

ax3.set_title(r'KL vs Variance Reduction', fontsize=14, pad=20)
ax3.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
# 限制y轴范围，避免被极端值拉伸
z_min, z_max = Z.min(), Z.max()
padding = (z_max - z_min) * 0.05

cbar = plt.colorbar(scatter, ax=ax3, shrink=0.6, pad=0.05)
cbar.set_label('|a-b|', rotation=270, labelpad=15)
# 添加红色虚线段 y=0，x从0到2
ax3.plot([0, 2.7], [0, 0], 'r--', linewidth=1.5, label=r'$\Delta$Var = 0')
ax3.legend(fontsize=12)
###################################### 图 4 ######################################
# ax4.set_zlim(0,100)
ax3.set_ylim(z_min - padding, z_max + padding)
# ax3.set_xlim(0, 2.5)
# ax3.set_ylim(-5, 40)
ax4.view_init(elev=20, azim=30)  # 优化视角

# 调整布局
plt.subplots_adjust(left=0.08, right=0.94, top=0.9, bottom=0.15, wspace=0.4)
# plt.savefig("./mois_figs/KL_VR_Eqq_under_normal.pdf", bbox_inches='tight')
plt.show()