import copy
import math
import random
import numpy as np


def function(x, row, col, mode='bidirectional', delta=0.1, power=1, bias=0, base=math.e,
             rr=(0, 1), exp=2, th=0.5, upper=1, lower=0, scale_factor=1, frequency=30,
             amplitude=1, mean=1, stddev=1, offset=1, constant=0, custom=None,
             percentage=None,qty=None, vector=None, mask_type=None):
    
    temp = x.copy()
    temp2 = False
    if mode == 'bidirectional':
        temp2 = x.copy()
        temp[0, row, col, 0] = (temp[0, row, col, 0] + bias) * (1 - delta * power)
        temp2[0, row, col, 0] = (temp2[0, row, col, 0] + bias) * (1 - delta * (-1) * power)

    elif mode == 'positive':
        temp[0, row, col, 0] = (temp[0, row, col, 0] + bias) * (1 + delta)

    elif mode ==  'negative':
        temp[0, row, col, 0] = (temp[0, row, col, 0] + bias) * (1 - delta)

    elif mode == 'bias':
        temp[0, row, col, 0] = temp[0, row, col, 0] + bias

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

    elif mode == 'upper-threshold':
        if temp[0, row, col, 0] <= th:
            temp[0, row, col, 0] = constant

    elif mode == 'lower-threshold':
        if temp[0, row, col, 0] >= th:
            temp[0, row, col, 0] = constant

    else:
        raise ValueError(f"Method not recognized, please choose one of from the list or use the\
                        default: %s" % str(['bidirectional', 'positive', 'negative', 'bias','log', 'random', 'exp',
                        'binary', 'sinusoidal', 'gaussian', 'tanh', 'sigmoid', 'relu', 'softplus',
                        'inverse', 'gaussian_noise', 'uniform_noise', 'offset', 'constant',
                        'vector', 'upper-threshold', 'lower-threshold']))

    return temp, temp2


