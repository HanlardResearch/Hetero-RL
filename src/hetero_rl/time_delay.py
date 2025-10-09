import numpy as np
from scipy import stats
import numpy as np
# import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings


class WeibullSampler:
    def __init__(self, lower_bound=60, upper_bound=1920, confidence=0.995, default=60,seed=42):
        """
        初始化威布尔分布采样器
        :param lower_bound: 置信区间下限（秒）
        :param upper_bound: 置信区间上限（秒）
        :param confidence: 置信度 (0-1之间)
        """
        self.lower = lower_bound
        self.upper = upper_bound
        self.confidence = confidence
        self.default = default
        if seed is not None:
            np.random.seed(seed)  # 设置全局随机种子以保证可重复性
        # 计算分布参数
        self.shape, self.scale = self.calculate_parameters()

        # 验证参数有效性
        self.validate_parameters()

    def calculate_parameters(self):
        """计算威布尔分布的形状参数和尺度参数"""
        # 计算分位点
        alpha = (1 - self.confidence) / 2
        lower_quantile = alpha
        upper_quantile = 1 - alpha

        # 定义方程组
        def equations(p):
            k, λ = p
            eq1 = 1 - np.exp(-(self.lower / λ) ** k) - lower_quantile
            eq2 = 1 - np.exp(-(self.upper / λ) ** k) - upper_quantile
            return [eq1, eq2]

        # 初始猜测值 (k, λ)
        # k通常0.5-5之间，λ取区间中值
        initial_guess = (1.0, (self.lower + self.upper) / 2)

        # 求解方程组
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k, λ = fsolve(equations, initial_guess)

        return k, λ

    def validate_parameters(self):
        """验证参数有效性"""
        if self.shape <= 0 or self.scale <= 0:
            raise ValueError("计算出的参数无效，请检查输入区间和置信度")

        # 计算实际置信区间
        actual_lower = self.scale * (-np.log(1 - (1 - self.confidence) / 2)) ** (1 / self.shape)
        actual_upper = self.scale * (-np.log((1 - self.confidence) / 2)) ** (1 / self.shape)

        # 检查是否匹配
        lower_diff = abs(actual_lower - self.lower) / self.lower
        upper_diff = abs(actual_upper - self.upper) / self.upper

        if lower_diff > 0.05 or upper_diff > 0.05:
            print(f"警告: 实际分位数与目标有偏差 (下界:{lower_diff * 100:.1f}%, 上界:{upper_diff * 100:.1f}%)")

    def sample(self, size=None):
        """
        生成服从威布尔分布的随机样本
        :param size: 样本数量 (None表示单个样本)
        :return: 随机样本
        """
        return self.scale * np.random.weibull(self.shape, size)

    def get_delay(self):

        T = self.sample(1)[0]
        return min(max(T - self.default, self.lower), self.upper)

    def pdf(self, t):
        """计算概率密度函数值"""
        return (self.shape / self.scale) * (t / self.scale) ** (self.shape - 1) * np.exp(
            -(t / self.scale) ** self.shape)

    def cdf(self, t):
        """计算累积分布函数值"""
        return 1 - np.exp(-(t / self.scale) ** self.shape)
    #
    # def plot_distribution(self, samples):
    #     """绘制分布图"""
    #     # 生成样本
    #     plt.figure(figsize=(12, 8))
    #
    #     # 直方图
    #     plt.subplot(2, 1, 1)
    #     hist, bins, _ = plt.hist(samples, bins=100, density=True, alpha=0.7, color='skyblue')
    #     plt.title(f'Weibull Distribution (k={self.shape:.3f}, λ={self.scale:.3f})')
    #     plt.xlabel('Delay Time (seconds)')
    #     plt.ylabel('Probability Density')
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #
    #     # 添加置信区间标记
    #     plt.axvline(x=self.lower, color='red', linestyle='--', label=f'Lower bound ({self.lower}s)')
    #     plt.axvline(x=self.upper, color='green', linestyle='--', label=f'Upper bound ({self.upper}s)')
    #     plt.fill_betweenx([0, max(hist)], self.lower, self.upper, color='yellow', alpha=0.2,
    #                       label=f'{self.confidence * 100:.1f}% Confidence Interval')
    #
    #     # 添加理论PDF曲线
    #     t = np.linspace(max(1, self.lower / 10), min(self.upper * 2, np.max(samples)), 1000)
    #     pdf_vals = self.pdf(t)
    #     plt.plot(t, pdf_vals, 'r-', linewidth=2, label='PDF')
    #
    #     plt.legend()
    #     plt.xlim(0, self.upper * 1.5)
    #
    #     # 累积分布函数图
    #     plt.subplot(2, 1, 2)
    #     sorted_samples = np.sort(samples)
    #     cdf_vals = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    #     plt.plot(sorted_samples, cdf_vals, 'b-', linewidth=1, label='Empirical CDF')
    #
    #     # 理论CDF
    #     theoretical_cdf = self.cdf(t)
    #     plt.plot(t, theoretical_cdf, 'r--', label='Theoretical CDF')
    #
    #     # 标记置信区间
    #     plt.axvline(x=self.lower, color='red', linestyle='--')
    #     plt.axvline(x=self.upper, color='green', linestyle='--')
    #     plt.axhline(y=(1 - self.confidence) / 2, color='gray', linestyle=':')
    #     plt.axhline(y=1 - (1 - self.confidence) / 2, color='gray', linestyle=':')
    #     plt.text(self.lower, 0.05, f'{(1 - self.confidence) / 2 * 100:.1f}%', fontsize=10, ha='right')
    #     plt.text(self.upper, 0.95, f'{(1 - (1 - self.confidence) / 2) * 100:.1f}%', fontsize=10)
    #
    #     plt.title('Cumulative Distribution Function (CDF)')
    #     plt.xlabel('Delay Time (seconds)')
    #     plt.ylabel('Cumulative Probability')
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.legend()
    #     plt.xlim(0, self.upper * 1.5)
    #
    #     plt.tight_layout()
    #     plt.show()
    #
    #     return samples

    def get_delay_list(self, n=500):
        delay_list = [self.get_delay()  for _ in range(n)]
        return delay_list


class LogNormalSampler:
    def __init__(self, lower_bound=60, upper_bound=1920, confidence=0.995, default=60, seed=42):
        """
        初始化威布尔分布采样器
        :param lower_bound: 置信区间下限（秒）
        :param upper_bound: 置信区间上限（秒）
        :param confidence: 置信度 (0-1之间)
        """
        self.lower = lower_bound
        self.upper = upper_bound
        self.confidence = confidence
        self.default = default
        if seed is not None:
            np.random.seed(seed)  # 设置全局随机种子以保证可重复性
    def get_delay(self, ):
        """
        生成服从对数正态分布的延迟时间样本，95%的样本落在60秒到1920秒之间

        参数:
            num_samples (int): 需要生成的样本数量

        返回:
            ndarray: 包含延迟时间(秒)的数组
        """
        # 定义目标区间和置信水平

        # 计算对应的正态分布参数
        # 对于对数正态分布，95%区间对应正态分布的μ±1.96σ
        z_score = stats.norm.ppf(1 - (1 - self.confidence) / 2)  # 95%置信的Z值(≈1.96)

        # 求解对数正态分布的参数
        # 方程组:
        #   ln(lower_bound) = μ - z_score * σ
        #   ln(upper_bound) = μ + z_score * σ
        A = np.array([[1, -z_score],
                      [1, z_score]])
        b = np.array([np.log(self.lower),
                      np.log(self.upper)])

        # 解线性方程组
        mu, sigma = np.linalg.solve(A, b)

        # 生成对数正态分布样本
        T = np.random.lognormal(mean=mu, sigma=sigma, size=1)[0]

        return min(max(T - self.default, self.lower), self.upper)

    def get_delay_list(self, n=500):
        delay_list = [self.get_delay()  for _ in range(n)]
        return delay_list




def get_delay_sampler(script_args):
    SAMPLER_FUNCS_REGISTRY = {
        "lognormal": LogNormalSampler,
        "weibull": WeibullSampler,
    }
    assert script_args.delay_sampler in SAMPLER_FUNCS_REGISTRY
    delay_sampler = SAMPLER_FUNCS_REGISTRY[script_args.delay_sampler]

    return delay_sampler(lower_bound=script_args.lower_bound,
                         upper_bound=script_args.upper_bound,
                         confidence=script_args.confidence,
                         default=script_args.default_delay,
                         )

class NoDelaySampler:
    def __init__(self, lower_bound=60, upper_bound=1920, confidence=0.995, default=60, seed=42):
        """
        初始化威布尔分布采样器
        :param lower_bound: 置信区间下限（秒）
        :param upper_bound: 置信区间上限（秒）
        :param confidence: 置信度 (0-1之间)
        """
        self.lower = lower_bound
        self.upper = upper_bound
        self.confidence = confidence
        self.default = default
        if seed is not None:
            np.random.seed(seed)  # 设置全局随机种子以保证可重复性
    def get_delay(self, ):
        return 0.0

    def get_delay_list(self, n=500):
        delay_list = [self.get_delay()  for _ in range(n)]
        return delay_list




def get_delay_sampler(script_args):
    SAMPLER_FUNCS_REGISTRY = {
        "lognormal": LogNormalSampler,
        "weibull": WeibullSampler,
        "nodelay": NoDelaySampler,
    }
    assert script_args.delay_sampler in SAMPLER_FUNCS_REGISTRY
    delay_sampler = SAMPLER_FUNCS_REGISTRY[script_args.delay_sampler]

    return delay_sampler(lower_bound=script_args.lower_bound,
                         upper_bound=script_args.upper_bound,
                         confidence=script_args.confidence,
                         default=script_args.default_delay,
                         )

#
# if __name__ == "__main__":
#     # 创建采样器 (60-1920秒, 95%置信区间)
#     sampler = WeibullSampler(lower_bound=60, upper_bound=1920, confidence=0.995)
#
#     print(f"计算参数: shape(k)={sampler.shape:.4f}, scale(λ)={sampler.scale:.4f}")
#
#     # 生成单个样本
#     print(f"单个样本: {sampler.sample():.2f}秒")
#
#     # 生成多个样本
#     samples = sampler.sample(10)
#     print(f"10个样本: {np.array2string(samples, precision=2, suppress_small=True)}")
#
#     # 验证置信区间
#     num_samples = 100000
#     validation_samples = sampler.sample(num_samples)
#     in_interval = np.sum((sampler.lower <= validation_samples) & (validation_samples <= sampler.upper))
#     percentage = in_interval / num_samples * 100
#
#     print(f"\n验证 ({num_samples}个样本):")
#     print(f"落在{sampler.lower}-{sampler.upper}秒的比例: {percentage:.2f}%")
#     print(f"理论置信度: {sampler.confidence * 100:.1f}%")
#     print(f"最小值: {np.min(validation_samples):.2f}秒")
#     print(f"最大值: {np.max(validation_samples):.2f}秒")
#     print(f"中位数: {np.median(validation_samples):.2f}秒")
#
#     # 绘制分布图
#     print("\n绘制分布图...")
#     sampler.plot_distribution()
# sampler = LogNormalSampler()
#
# sampler.get_delay()
# print("lognormal", sampler.get_delay())
#
# sampler2 = WeibullSampler()
# print("Weibull", sampler2.get_delay())