import math, random
import numpy as np


class Bidirectional:
    """
    Calculates perturbations for positive and negative changes.

    :type delta: float
    :param delta: Percetage of positive and negative change as 1+delta or 1-delta

    :rtype: 4d array
    :return: Both of perturbated arrays
    """
    def __init__(self, delta=0.1):
        self.delta = delta
    def apply(self, x, row, col, window, bias):
        x2 = x.copy()
        x[0, row:row+window[0], col:col+window[1], 0] = (x[0, row:row+window[0], col:col+window[1], 0] + bias) * (1 + self.delta)
        x2[0, row:row+window[0], col:col+window[1], 0] = (x2[0, row:row+window[0], col:col+window[1], 0] + bias) * (1 - self.delta)
        return x, x2


class Positive:
    """
    Applies a positive perturbation to the input array.

    :type delta: float
    :param delta: Percentage of positive change as 1+delta

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, delta=0.1):
        self.delta = delta
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = (x[0, row:row+window[0], col:col+window[1], 0] + bias) * (1 + self.delta)
        return x, None


class Negative:
    """
    Applies a negative perturbation to the input array.

    :type delta: float
    :param delta: Percentage of negative change as 1-delta

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, delta=0.1):
        self.delta = delta
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = (x[0, row:row+window[0], col:col+window[1], 0] + bias) * (1 - self.delta)
        return x, None   


class Log:
    """
    Applies a logarithmic perturbation to the input array.

    :type base: int
    :param base: The base of the logarithm.

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, base=10):
        self.base = base
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = math.log(x[0, row:row+window[0], col:col+window[1], 0] + bias, self.base)
        return x, None


class RandomPtn:
    """
    Applies a random perturbation to the input array.

    :type r: tuple
    :param r: The range of random values for perturbation.

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, r=(0, 1)):
        self.r = r
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = (x[0, row:row+window[0], col:col+window[1], 0] + bias) * random.uniform(self.r[0], self.r[1])
        return x, None


class Exp:
    """
    Applies an exponential perturbation to the input array.

    :type exp: float
    :param exp: The exponent for the perturbation.

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, exp=2):
        self.exp = exp
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = (x[0, row:row+window[0], col:col+window[1], 0] + bias) ** self.exp
        return x, None


class Binary:
    """
    Applies a binary perturbation to the input array based on a threshold.

    :type th: float
    :param th: Threshold value for binary perturbation.

    :type upper: float
    :param upper: Value to set if input is above the threshold.

    :type lower: float
    :param lower: Value to set if input is below the threshold.

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, th=0, upper=1, lower=0):
        self.th = th
        self.upper = upper
        self.lower = lower
    def apply(self, x, row, col, window, bias):
        if np.all(x[0, row, col, 0]) > self.th:
            x[0, row:row+window[0], col:col+window[1], 0] = self.upper
        else:
            x[0, row:row+window[0], col:col+window[1], 0] = self.lower
        return x, None


class Sinusoidal:
    """
    Applies a sinusoidal perturbation to the input array.

    :type freq: float
    :param freq: Frequency of the sine wave.

    :type amp: float
    :param amp: Amplitude of the sine wave.

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, freq=30, amp=1):
        self.freq = freq
        self.amp = amp
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = (x[0, row:row+window[0], col:col+window[1], 0] + bias) * math.sin(self.freq * x[0, row:row+window[0], col:col+window[1], 0]) + self.amp
        return x, None
    

class Gaussian:
    """
    Applies a Gaussian perturbation to the input array.

    :type mean: float
    :param mean: Mean of the Gaussian distribution.

    :type stdv: float
    :param stdv: Standard deviation of the Gaussian distribution.

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, mean=1, stdv=1):
        self.mean = mean
        self.stdv = stdv
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = (x[0, row:row+window[0], col:col+window[1], 0] + bias) * math.exp(-((x[0, row:row+window[0], col:col+window[1], 0] - self.mean) ** 2) / (2 * self.stdv ** 2))
        return x, None


class Tanh:
    """
    Applies a hyperbolic tangent perturbation to the input array.

    :type sf: float
    :param sf: Scale factor to apply before the tanh function.

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, sf=1):
        self.sf = sf # scale factor
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = math.tanh(self.sf * (x[0, row:row+window[0], col:col+window[1], 0] + bias))
        return x, None


class Sigmoid:
    """
    Applies a sigmoid perturbation to the input array.

    :type sf: float
    :param sf: Scale factor to apply before the sigmoid function.

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, sf=1):
        self.sf = sf
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = 1 / (1 + math.exp(-self.sf * (x[0, row:row+window[0], col:col+window[1], 0] + bias)))
        return x, None
    

class ReLU:
    """
    Applies a Rectified Linear Unit (ReLU) perturbation to the input array.

    :rtype: 4d array
    :return: Perturbated array
    """
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = max(0, x[0, row:row+window[0], col:col+window[1], 0] + bias)
        return x, None
    

class LeakyReLU:
    """
    Applies a Leaky Rectified Linear Unit (LeakyReLU) perturbation to the input array.

    :type alpha: float
    :param alpha: Coefficient for negative inputs in LeakyReLU.

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, alpha=1):
        self.alpha = alpha
    def apply(self, x, row, col, window, bias):
        if x[0, row, col, 0] < 0:
            x[0, row:row+window[0], col:col+window[1], 0] = (x[0, row:row+window[0], col:col+window[1], 0] + bias) * self.alpha
        return x, None
    

class Softplus:
    """
    Applies a Softplus perturbation to the input array.

    :type sf: float
    :param sf: Scale factor to apply before the Softplus function.

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, sf=1):
        self.sf = sf
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = math.log(1 + math.exp(self.sf * (x[0, row:row+window[0], col:col+window[1], 0] + bias)))
        return x, None
    

class Inverse:
    """
    Applies an inverse perturbation to the input array.

    :rtype: 4d array
    :return: Perturbated array
    """
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = (1 / (x[0, row:row+window[0], col:col+window[1], 0] + bias))
        return x, None


class GaussianNoise:
    """
    Applies Gaussian noise to the input array.

    :type mean: float
    :param mean: Mean of the Gaussian distribution.

    :type stdv: float
    :param stdv: Standard deviation of the Gaussian distribution.

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, mean=1, stdv=1):
        self.mean = mean
        self.stdv = stdv
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = x[0, row:row+window[0], col:col+window[1], 0] + bias + random.gauss(self.mean, self.stdv)
        return x, None


class UniformNoise:
    """
    Applies uniform noise to the input array.

    :type lower: float
    :param lower: Lower bound of the uniform distribution.

    :type upper: float
    :param upper: Upper bound of the uniform distribution.

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, lower=-1, upper=1):
        self.upper = upper
        self.lower = lower
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = x[0, row:row+window[0], col:col+window[1], 0] + bias + random.uniform(self.lower, self.upper)
        return x, None


class Offset:
    """
    Applies an offset to the input array.

    :type offset: float
    :param offset: The value to be added to the input array.

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, offset=1):
        self.offset = offset
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = x[0, row:row+window[0], col:col+window[1], 0] + bias + self.offset
        return x, None


class Constant:
    """
    Replaces all values in the input array with a constant.

    :type c: float
    :param c: The constant value.

    :rtype: 4d array
    :return: Perturbated array
    """
    def __init__(self, c=0):
        self.c = c
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = self.c
        return x, None


class Custom:
    """
    Applies a custom function as a perturbation to the array.

    :type func: callable
    :param func: A function that takes a single argument and returns a single value.

    :rtype: 4d array
    :return: Custom perturbated array
    """
    def __init__(self, custom):
        self.custom = custom
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = x[0, row:row+window[0], col:col+window[1], 0]*self.custom[0, row:row+window[0], col:col+window[1], 0]
        return x, None


class CustomSum:
    """
    Sums a custom vector as a perturbation to the array.

    :type func: callable
    :param func: A function that takes a single argument and returns a single value.

    :rtype: 4d array
    :return: Custom perturbated array
    """
    def __init__(self, custom):
        self.custom = custom
    def apply(self, x, row, col, window, bias):
        x[0, row:row+window[0], col:col+window[1], 0] = x[0, row:row+window[0], col:col+window[1], 0] + self.custom[0, row:row+window[0], col:col+window[1], 0]
        return x, None


class UpperThreshold:
    """
    Sets values above a certain threshold to a specified value.

    :type threshold: float
    :param threshold: Threshold above which values will be set to a specific value.
    :type value: float
    :param value: Value to which entries above the threshold will be set.

    :rtype: 4d array
    :return: Array with upper threshold applied
    """
    def __init__(self, th=0, c=0):
        self.th = th
        self.c = c
    def apply(self, x, row, col, window, bias):
        if x[0, row, col, 0] <= self.th:
            x[0, row:row+window[0], col:col+window[1], 0] = self.c
        return x, None


class LowerThreshold:
    """
    Sets values below a certain threshold to a specified value.

    :type threshold: float
    :param threshold: Threshold below which values will be set to a specific value.
    :type value: float
    :param value: Value to which entries below the threshold will be set.

    :rtype: 4d array
    :return: Array with lower threshold applied
    """
    def __init__(self, th=0, c=0):
        self.th = th
        self.c = c
    def apply(self, x, row, col, window, bias):
        if x[0, row, col, 0] >= self.th:
            x[0, row:row+window[0], col:col+window[1], 0] = self.c
        return x, None
