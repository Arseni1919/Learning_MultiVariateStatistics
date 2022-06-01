import random
from scipy.optimize import fsolve
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt


def q1_gimel_func(c_values):
    return_value = []
    # n equations of equality
    for i in range(1, N + 1):
        equation = (lambda_0 - lambda_1) * c_values[0] - (i * np.log((p_1 * (1 - p_0)) / (p_0 * (1 - p_1)))) - (lambda_0 - lambda_1) * c_values[i]
        return_value.append(equation)
    # n+1 equation of alpha
    curr_sum = 0
    for i in range(N + 1):
        i_binom = binom.pmf(i, N, p_0)
        i_gamma = 0
        for j in range(N + 1):
            i_gamma += np.exp(-lambda_0 * c_values[i]) * ((lambda_0 * c_values[i]) ** j) / np.math.factorial(j)
        curr_sum += i_binom * (1 - i_gamma)
    curr_sum -= alpha
    return_value.append(curr_sum)
    return return_value


def targil_1_seif_gimel():
    # start = np.array(np.zeros(N + 1))
    start = np.random.rand(N + 1)
    root = fsolve(q1_gimel_func, start)
    print(root)
    plt.scatter(list(range(N + 1)), root)
    plt.show()

    # check:
    zeros_list = np.zeros(N + 1)
    out1 = np.isclose(q1_gimel_func(root), zeros_list)  # func(root) should be almost 0.0.
    print(out1)


def targil_1_seif_dalet():
    pass


def main():
    # set seed
    if to_use_seed:
        random.seed(seed)
        np.random.seed(seed)

    targil_1_seif_gimel()
    targil_1_seif_dalet()


if __name__ == '__main__':
    seed = 123
    # to_use_seed = True
    to_use_seed = False
    N = 100
    alpha = 0.05
    p_0, lambda_0 = 0.6, 1.0
    p_1, lambda_1 = 0.57, 0.95
    main()
















# def func(x):
#     return_value = []
#     for i in [0, 1]:
#         if i == 0:
#             return_value.append(x[0] * np.cos(x[1]) - 4)
#         if i == 1:
#             return_value.append(x[1] * x[0] - x[1] - 5)
#     return return_value
