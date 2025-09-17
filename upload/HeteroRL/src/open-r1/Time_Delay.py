import numpy as np
from scipy import stats
import numpy as np
# import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings


class WeibullSampler:
    def __init__(self, lower_bound=60, upper_bound=1920, confidence=0.995, default=60,seed=42):

        self.lower = lower_bound
        self.upper = upper_bound
        self.confidence = confidence
        self.default = default
        if seed is not None:
            np.random.seed(seed)

        self.shape, self.scale = self.calculate_parameters()
        self.validate_parameters()

    def calculate_parameters(self):

        alpha = (1 - self.confidence) / 2
        lower_quantile = alpha
        upper_quantile = 1 - alpha

        def equations(p):
            k, λ = p
            eq1 = 1 - np.exp(-(self.lower / λ) ** k) - lower_quantile
            eq2 = 1 - np.exp(-(self.upper / λ) ** k) - upper_quantile
            return [eq1, eq2]

        initial_guess = (1.0, (self.lower + self.upper) / 2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k, λ = fsolve(equations, initial_guess)

        return k, λ

    def validate_parameters(self):
        if self.shape <= 0 or self.scale <= 0:
            raise ValueError("计算出的参数无效，请检查输入区间和置信度")

        actual_lower = self.scale * (-np.log(1 - (1 - self.confidence) / 2)) ** (1 / self.shape)
        actual_upper = self.scale * (-np.log((1 - self.confidence) / 2)) ** (1 / self.shape)

        lower_diff = abs(actual_lower - self.lower) / self.lower
        upper_diff = abs(actual_upper - self.upper) / self.upper

        if lower_diff > 0.05 or upper_diff > 0.05:
            print(f"警告: 实际分位数与目标有偏差 (下界:{lower_diff * 100:.1f}%, 上界:{upper_diff * 100:.1f}%)")

    def sample(self, size=None):
        return self.scale * np.random.weibull(self.shape, size)

    def get_delay(self):

        T = self.sample(1)[0]
        return min(max(T - self.default, self.lower), self.upper)

    def pdf(self, t):
        return (self.shape / self.scale) * (t / self.scale) ** (self.shape - 1) * np.exp(
            -(t / self.scale) ** self.shape)

    def cdf(self, t):
        return 1 - np.exp(-(t / self.scale) ** self.shape)

    def get_delay_list(self, n=500):
        delay_list = [self.get_delay()  for _ in range(n)]
        return delay_list


class LogNormalSampler:
    def __init__(self, lower_bound=60, upper_bound=1920, confidence=0.995, default=60, seed=42):
        self.lower = lower_bound
        self.upper = upper_bound
        self.confidence = confidence
        self.default = default
        if seed is not None:
            np.random.seed(seed)

    def get_delay(self, ):
        z_score = stats.norm.ppf(1 - (1 - self.confidence) / 2)
        A = np.array([[1, -z_score],
                      [1, z_score]])
        b = np.array([np.log(self.lower),
                      np.log(self.upper)])
        mu, sigma = np.linalg.solve(A, b)
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
        self.lower = lower_bound
        self.upper = upper_bound
        self.confidence = confidence
        self.default = default
        if seed is not None:
            np.random.seed(seed)
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
