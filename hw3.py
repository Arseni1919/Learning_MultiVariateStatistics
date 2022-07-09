import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def q1_PCA_func():
    print('=========== ========== ===========')
    print('=========== ========== ===========')
    print('=========== ========== ===========')
    N_iterations = 100
    A = np.array(
        [
            [1, .2, .4, .36],
            [.2, 1, .3, .27],
            [.4, .3, 1, .4],
            [.36, .27, .4, 1]
        ]
    )

    h_squared = np.max(A - np.identity(4), axis=1)
    for i in range(N_iterations):
        psi_i = 1 - h_squared
        Psi = np.identity(4)
        for j in range(4):
            Psi[j][j] = psi_i[j]
        # print(h_squared)
        Lambda_Squared = A - Psi
        w, v = np.linalg.eig(Lambda_Squared)
        eigen_val_1 = w[0]
        eigen_vector_1 = v[0]
        Lambda = np.array([np.sqrt(eigen_val_1) * eigen_vector_1])
        h_squared = np.sum(np.power(Lambda, 2), 0)

        # Print
        if i > N_iterations - 3:
            print('----')
            print(f'eigen_val_1: {eigen_val_1}')
            print(f'eigen_vector_1: {eigen_vector_1}')
            print(f'Lambda: {Lambda}')
            print(f'Psi: {Psi}')
    """
    Last two iterations:
    ----
    eigen_val_1: 1.366199543761684
    eigen_vector_1: [ 0.46981363  0.47237152 -0.58975003 -0.45643751]
    Lambda: [[ 0.54913949  0.55212927 -0.68932661 -0.53350488]]
    Psi: [[0.7034894  0.         0.         0.        ]
         [0.         0.82715312 0.         0.        ]
         [0.         0.         0.35031741 0.        ]
         [0.         0.         0.         0.78443174]]
    ----
    eigen_val_1: 1.3319694265665667
    eigen_vector_1: [ 0.48386032  0.42454334 -0.62076603  0.44753959]
    Lambda: [[ 0.55842794  0.48996963 -0.71643216  0.51650982]]
    Psi: [[0.69844582 0.         0.         0.        ]
         [0.         0.69515326 0.         0.        ]
         [0.         0.         0.52482883 0.        ]
         [0.         0.         0.         0.71537254]]
    """


def q1_MLE_func():
    print('=========== ========== ===========')
    print('=========== ========== ===========')
    print('=========== ========== ===========')
    N_iterations = 100
    A = np.array(
        [
            [1, .2, .4, .36],
            [.2, 1, .3, .27],
            [.4, .3, 1, .4],
            [.36, .27, .4, 1]
        ]
    )

    h_squared = np.max(A - np.identity(4), axis=1)
    for i in range(N_iterations):
        pass

        # Print
        if i > N_iterations - 3:
            print('----')
            # print(f'eigen_val_1: {eigen_val_1}')
            # print(f'eigen_vector_1: {eigen_vector_1}')
            # print(f'Lambda: {Lambda}')
            # print(f'Psi: {Psi}')
    """
    Last two iterations:

    """


def q2_func():
    print('=========== ========== ===========')
    print('=========== ========== ===========')
    print('=========== ========== ===========')
    N_iterations = 100
    alpha_list = []
    l_1_list = []
    l_0_list = []

    alpha = random.random()
    lambda_0 = 1 + random.random() * 2
    lambda_1 = 1 + random.random() * 2

    y_data = pd.read_excel('3-2.xlsx', header=None)
    y_data = y_data[0].to_numpy()
    data_N = len(y_data)
    for i in range(N_iterations):

        # E step
        I_star_list = []
        for y_i in y_data:
            I_star_i = (alpha*lambda_1 * np.exp(-lambda_1 * y_i))/(alpha*lambda_1 * np.exp(-lambda_1 * y_i) + (1-alpha) * lambda_0 * np.exp(-lambda_0*y_i))
            I_star_list.append(I_star_i)
        I_star_list = np.array(I_star_list)

        # M step
        alpha = sum(I_star_list) / data_N
        lambda_0 = sum(I_star_list) / sum(I_star_list * y_data)
        lambda_1 = (data_N - sum(y_data)) / (sum(y_data * (1 - I_star_list)))

        # Print
        if i > N_iterations - 3:
            print(f'[{i + 1}]: alpha: {alpha : .4f}, lambda 0: {lambda_0 : .4f}, lambda 1: {lambda_1 : .4f}')

        alpha_list.append(alpha)
        l_0_list.append(lambda_0)
        l_1_list.append(lambda_1)

    # Plot results
    plt.plot(range(len(alpha_list)), alpha_list, label='alpha')
    plt.plot(range(len(l_0_list)), l_0_list, label='lambda 0')
    plt.plot(range(len(l_1_list)), l_1_list, label='lambda 1')
    plt.legend()
    plt.show()
    """
    Last two iterations:
    [99]: alpha:  0.1137, lambda 0:  1.6329, lambda 1:  0.5922
    [100]: alpha:  0.1076, lambda 0:  0.6391, lambda 1:  0.7126
    """


def main():
    q1_PCA_func()
    q1_MLE_func()
    q2_func()


if __name__ == '__main__':
    main()
