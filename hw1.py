import random
from scipy.optimize import fsolve
from scipy.stats import binom
import numpy as np
# import matplotlib.pyplot as plt


def q1_pdf_func(N, p_k, lambda_k, c_values):
    curr_sum = 0
    for i in range(N + 1):
        i_binom = binom.pmf(i, N, p_k)
        i_gamma = 0
        for j in range(N):
            i_gamma += np.exp(-lambda_k * c_values[i]) * ((lambda_k * c_values[i]) ** j) / np.math.factorial(j)
        # curr_sum += i_binom * (1 - i_gamma)
        curr_sum += i_binom * i_gamma
    return curr_sum


def q1_equations_func(c_values):
    alpha = 0.05
    q1_p_0, q1_lambda_0 = 0.6, 1.0
    # RANDOM P AND LAMBDA VALUES
    q1_p_1, q1_lambda_1 = 0.47, 0.85

    N = len(c_values) - 1

    return_value = []

    # n equations of equality
    for i in range(1, N + 1):
        equation = (q1_lambda_0 - q1_lambda_1) * c_values[0] - (
                i * np.log((q1_p_1 * (1 - q1_p_0)) / (q1_p_0 * (1 - q1_p_1)))) - (q1_lambda_0 - q1_lambda_1) * \
                   c_values[i]
        return_value.append(equation)

    # n+1 equation of alpha
    curr_sum = q1_pdf_func(N, q1_p_0, q1_lambda_0, c_values)
    curr_sum -= alpha
    return_value.append(curr_sum)

    return return_value


def q_1_seif_gimel(to_plot=True):
    if to_plot:
        print('=========== Q1 - SEIF GIMEL ===========')
    N = 100

    # start = np.array(np.zeros(N + 1))
    start = np.random.rand(N + 1)
    if to_plot:
        print('Start solving...')
    c_values = fsolve(q1_equations_func, start)
    if to_plot:
        print('Finished to solve.')

    # check:
    zeros_list = np.zeros(N + 1)
    out1 = sum((q1_equations_func(c_values) - zeros_list) ** 2)  # func(c_values) should be almost 0.0.

    # print & plot results
    if to_plot:
        print(f'The error: {out1 : .2f}')
        print(f'C values: {c_values}')
        # plt.scatter(list(range(N + 1)), c_values)
        # plt.show()
        print('=========== ========== ===========')
    return c_values, N


def q_1_seif_dalet(c_values=None, N=None):
    q1_p_0, q1_lambda_0 = 0.6, 1.0
    # RANDOM P AND LAMBDA VALUES
    q1_p_1, q1_lambda_1 = 0.47, 0.85

    if c_values is None:
        print('Start solving...')
        c_values, N = q_1_seif_gimel(to_plot=False)
        print('Finished to solve.')

    curr_sum_0 = q1_pdf_func(N, q1_p_0, q1_lambda_0, c_values)
    curr_sum_1 = q1_pdf_func(N, q1_p_1, q1_lambda_1, c_values)
    print(f'[Q1 - dalet] - P_0(H_1): {curr_sum_0}, P_1(H_1) (power): {curr_sum_1}')
    print('=========== ========== ===========')


def q2_log_likelihood(curr_p, curr_k, curr_s, curr_n):
    log_likelihood = (curr_k + curr_n) * np.log(curr_p) - curr_k * np.log(1 - curr_p) - (curr_p / (1 - curr_p)) * curr_s
    return log_likelihood


def q2_seif_bet():
    N = 200
    q2_p = 0.48
    q2_lambda = q2_p / (1 - q2_p)
    critic_chi_square_value = 3.841
    test_list = []
    for i in range(100):
        k = np.random.binomial(N, q2_p, 1)[0]
        s = np.random.gamma(N, 1 / q2_lambda, 1)[0]
        # s = sum([np.random.exponential(scale=1/q2_lambda) for _ in range(N)])
        p_hat = (k + 2 * N + s - np.sqrt((k + 2 * N + s) ** 2 - 4 * N * (k + N))) / (2 * N)
        # print(f'p_hat: {p_hat}, k: {k}, s:{s}')
        log_L = q2_log_likelihood(p_hat, k, s, N)
        log_L_0 = q2_log_likelihood(0.5, k, s, N)
        two_log_ratio = 2 * (log_L - log_L_0)
        decision = two_log_ratio > critic_chi_square_value
        test_list.append(decision)
    power = sum(test_list) / len(test_list)
    print(f'[Q2 - bet] - The power is: {power}')
    print('=========== ========== ===========')


def q3_get_wilks_decision(sample_1, sample_2, sample_3, sample_united, N, critic_chi_square_value):
    cov_1 = np.cov(sample_1.T)
    cov_2 = np.cov(sample_2.T)
    cov_3 = np.cov(sample_3.T)
    cov_united = np.cov(sample_united.T)

    det_1 = np.linalg.det(cov_1)
    det_2 = np.linalg.det(cov_2)
    det_3 = np.linalg.det(cov_3)
    det_united = np.linalg.det(cov_united)

    # likelihood_ratio = (det_1 ** (-N/2)) * (det_2 ** (-N/2)) * (det_3 ** (-N/2)) / (det_united ** (-N*3/2))
    log_likelihood_ratio = (-N / 2) * (np.log(det_1) + np.log(det_2) + np.log(det_3)) + (3 * N / 2) * np.log(det_united)
    # two_log_ratio = 2 * np.log(likelihood_ratio)
    two_log_ratio = 2 * log_likelihood_ratio
    decision = two_log_ratio > critic_chi_square_value
    return decision


def q3_seif_bet():
    sigma_from_file = np.array(
        [[8.345869685,  1.630590511,    2.068417519,    1.651517967,    -4.898867411],
         [1.63059051,   1.85486806,     1.30346045,     1.31463125,     -3.15375118],
         [2.06841752,   1.30346045,     5.49344116,     -0.25419136,    -1.11616971],
         [1.65151797,   1.31463125,     -0.25419136,    2.1713275,      -1.97311836],
         [-4.89886741, - 3.15375118,    -1.11616971,    -1.97311836,    7.42084205]]
    )
    critic_chi_square_value = 43.773
    N = 100
    test_list = []
    for _ in range(100):
        sample_1 = np.random.multivariate_normal(mean=np.zeros(5), cov=sigma_from_file, size=N)
        sample_2 = np.random.multivariate_normal(mean=np.zeros(5), cov=sigma_from_file, size=N)
        sample_3 = np.random.multivariate_normal(mean=np.zeros(5), cov=sigma_from_file, size=N)
        sample_united = np.concatenate((sample_1, sample_2, sample_3))

        decision = q3_get_wilks_decision(sample_1, sample_2, sample_3, sample_united, N, critic_chi_square_value)
        test_list.append(decision)

    power = sum(test_list) / len(test_list)
    print(f'[Q3 - bet] - The power is: {power}')
    print('=========== ========== ===========')


def q3_seif_gimel():
    critic_chi_square_value = 43.773
    N = 100
    test_list = []
    for _ in range(100):
        sample_1 = np.random.multivariate_normal(mean=np.zeros(5), cov=0.5 * np.identity(5), size=N)
        sample_2 = np.random.multivariate_normal(mean=np.zeros(5), cov=np.identity(5), size=N)
        sample_3 = np.random.multivariate_normal(mean=np.zeros(5), cov=1.5 * np.identity(5), size=N)
        sample_united = np.concatenate((sample_1, sample_2, sample_3))

        decision = q3_get_wilks_decision(sample_1, sample_2, sample_3, sample_united, N, critic_chi_square_value)
        test_list.append(decision)

    power = sum(test_list) / len(test_list)
    print(f'[Q3 - gimel] - The power is: {power}')
    print('=========== ========== ===========')


def main():
    # Q1
    """
    # Sometimes the solver does not work correctly because of the starting positions.
    # So it needed to execute the code once again. (Change the seed)
    """
    c_values, N = q_1_seif_gimel()
    q_1_seif_dalet(c_values, N)

    # Q2
    q2_seif_bet()

    # Q3
    q3_seif_bet()
    q3_seif_gimel()


if __name__ == '__main__':
    seed = 12
    to_use_seed = True
    random.seed(seed)
    np.random.seed(seed)

    main()

