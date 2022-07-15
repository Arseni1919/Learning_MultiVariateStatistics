import scipy as sci
import numpy as np
from scipy.optimize import minimize


def log_likelihood_func_fixed_psi(lambdas, *args):
    a = np.array([lambdas])
    b = np.array([lambdas])
    print(a * b.T)
    multiplied_lambdas_matrix = np.array([lambdas]) * np.array([lambdas])
    psi_matrix = np.diag([args[0], args[1], args[2], args[3]])
    sigma_matrix = multiplied_lambdas_matrix + psi_matrix
    return -100 * (np.log(np.linalg.det(sigma_matrix)) + np.trace(np.linalg.inv(sigma_matrix) * args[4]))


def main():
    params = np.array([.1, .2, .2, .2])
    res = minimize(log_likelihood_func_fixed_psi, params, args=(2, 3, 4, 5))
    print(res.x)
    print(res.fun)
    a = np.array([[1, 2, 3]])
    b = np.array([[1, 2, 3]])
    # print(np.cross(a, b.T))
    # print(a @ b.T)
    print(a * b.T)


if __name__ == '__main__':
    main()
