import copy
import math
import random
import numpy as np


def function(pf, x, row, col, mode='bidirectional', delta=0.1, power=1, 
             bias=0, base=math.e, rr=(0, 1), exp=2, th=0.5, upper=1, 
             lower=0, scale_factor=1, frequency=30, amplitude=1, mean=1, 
             stddev=1, offset=1, constant=1, custom=None):
    
    temp = x.copy()
    temp2 = False
    if mode == 'bidirectional':
        temp2 = x.copy()
        temp[0, row, col, 0] = (temp[0, row, col, 0] + bias) * (1 - delta * power)
        temp2[0, row, col, 0] = (temp2[0, row, col, 0] + bias) * (1 - delta * (-1) * power)

    elif mode in ['positive', 'negative']:
        factor = 1 + delta if mode == 'positive' else 1 - delta
        temp[0, row, col, 0] = (temp[0, row, col, 0] + bias) * factor

    elif mode == 'log':
        temp[0, row, col, 0] = math.log(temp[0, row, col, 0] + bias, base)

    elif mode == 'random':
        temp[0, row, col, 0] = (temp[0, row, col, 0] + bias) * random.uniform(rr[0], rr[1])

    elif mode == 'exp':
        temp[0, row, col, 0] = (temp[0, row, col, 0] + bias) ** exp

    elif mode == 'binary':
        if np.all(temp[0, row, col, 0]) > th:
            temp[0, row, col, 0] = upper # upper value
        else:
            temp[0, row, col, 0] = lower # assume default

    elif mode == 'sinusoidal':
        temp[0, row, col, 0] = (temp[0, row, col, 0] + bias) * math.sin(frequency * temp[0, row, col, 0]) + amplitude

    elif mode == 'gaussian':
        temp[0, row, col, 0] = (temp[0, row, col, 0] + bias) * math.exp(-((temp[0, row, col, 0] - mean) ** 2) / (2 * std_dev ** 2))

    elif mode == 'tanh':
        temp[0, row, col, 0] = math.tanh(scale_factor * (temp[0, row, col, 0] + bias))

    elif mode == 'sigmoid':
        temp[0, row, col, 0] = 1 / (1 + math.exp(-scale_factor * (temp[0, row, col, 0] + bias)))

    elif mode == 'relu':
        temp[0, row, col, 0] = max(0, temp[0, row, col, 0] + bias)

    elif mode == 'softplus':
        temp[0, row, col, 0] = math.log(1 + math.exp(scale_factor * (temp[0, row, col, 0] + bias)))

    elif mode == 'inverse':
        temp[0, row, col, 0] = (1 / (temp[0, row, col, 0] + bias))

    elif mode == 'gaussian_noise':
        temp[0, row, col, 0] = temp[0, row, col, 0] + random.gauss(mean, stddev)

    elif mode == 'uniform_noise':
        temp[0, row, col, 0] = temp[0, row, col, 0] + random.uniform(lower, upper)

    elif mode == 'offset':
        temp[0, row, col, 0] = temp[0, row, col, 0] + offset
    
    elif mode == 'constant':
        temp[0, row, col, 0] = constant

    elif mode == 'vector':
        temp[0, row, col, 0] = temp[0, row, col, 0]*custom[0, row, col, 0]

    return temp, temp2



# def function(pf, x, row, col, mode='bidirectional', delta=0.1, power=1, 
#              bias=0, absolute=False, base=math.e, rr=(0, 1), exp=2, th=0.5,
#              upper=1, lower=0, scale_factor=1, frequency=30, amplitude=1, 
#              mean=1, stddev=1, offset=1, constant=1, custom=None):
    
#     p0 = pf(x)
#     temp = x.copy()
    
#     if mode == 'bidirectional':
#         p = [p0, None, None]
#         for j in range(1, 3):
#             temp[0, row, col, 0] = temp[0, row, col, 0] * (1 - delta * (j * 2 - 3) * power) + bias
#             p[j] = pf(temp)
#         val = (p[2] + p[1]) / 2

#     elif mode in ['positive', 'negative']:
#         factor = 1 + delta if mode == 'positive' else 1 - delta
#         temp[0, row, col, 0] = temp[0, row, col, 0] * factor + bias
#         p1 = pf(temp)
#         val = p1 - p0

#     elif mode == 'log':
#         temp[0, row, col, 0] = math.log(temp[0, row, col, 0] + bias, base)
#         p1 = pf(temp)
#         val = p1 - p0

#     elif mode == 'random':
#         temp[0, row, col, 0] = (temp[0, row, col, 0] + bias) * random.uniform(rr[0], rr[1])
#         p1 = pf(temp)
#         val = p1 - p0

#     elif mode == 'exp':
#         temp[0, row, col, 0] = (temp[0, row, col, 0] + bias) ** exp
#         p1 = pf(temp)
#         val = p1 - p0

#     elif mode == 'binary':
#         val = lower # assume default
#         if np.all(temp[0, row, col, 0]) > th:
#             val = upper # upper value

#     elif mode == 'sinusoidal':
#         temp[0, row, col, 0] = (temp[0, row, col, 0] + bias) * math.sin(frequency * temp[0, row, col, 0]) + amplitude
#         p1 = pf(temp)
#         val = p1 - p0

#     elif mode == 'gaussian':
#         temp[0, row, col, 0] = (temp[0, row, col, 0] + bias) * math.exp(-((temp[0, row, col, 0] - mean) ** 2) / (2 * std_dev ** 2))
#         p1 = pf(temp)
#         val = p1 - p0

#     elif mode == 'tanh':
#         temp[0, row, col, 0] = math.tanh(scale_factor * (temp[0, row, col, 0] + bias))
#         p1 = pf(temp)
#         val = p1 - p0

#     elif mode == 'sigmoid':
#         temp[0, row, col, 0] = 1 / (1 + math.exp(-scale_factor * (temp[0, row, col, 0] + bias)))
#         p1 = pf(temp)
#         val = p1 - p0

#     elif mode == 'relu':
#         temp[0, row, col, 0] = max(0, temp[0, row, col, 0] + bias)
#         p1 = pf(temp)
#         val = p1 - p0

#     elif mode == 'softplus':
#         temp[0, row, col, 0] = math.log(1 + math.exp(scale_factor * (temp[0, row, col, 0] + bias)))
#         p1 = pf(temp)
#         val = p1 - p0

#     elif mode == 'inverse':
#         temp[0, row, col, 0] = (1 / (temp[0, row, col, 0] + bias))
#         p1 = pf(temp)
#         val = p1 - p0

#     elif mode == 'gaussian_noise':
#         temp[0, row, col, 0] = temp[0, row, col, 0] + random.gauss(mean, stddev)
#         p1 = pf(temp)
#         val = p1 - p0

#     elif mode == 'uniform_noise':
#         temp[0, row, col, 0] = temp[0, row, col, 0] + random.uniform(lower, upper)
#         p1 = pf(temp)
#         val = p1 - p0

#     elif mode == 'offset':
#         temp[0, row, col, 0] = temp[0, row, col, 0] + offset
#         p1 = pf(temp)
#         val = p1 - p0
    
#     elif mode == 'constant':
#         temp[0, row, col, 0] = constant
#         p1 = pf(temp)
#         val = p1 - p0

#     elif mode == 'custom':
#         temp[0, row, col, 0] = temp[0, row, col, 0]*custom[0, row, col, 0]
#         p1 = pf(temp)
#         val = p1 - p0

#     if absolute:
#         if isinstance(val, list):
#             return [abs(i) for i in val]
#         else:
#             return abs(val)

#     return val
