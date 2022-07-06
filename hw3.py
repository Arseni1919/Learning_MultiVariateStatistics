import numpy as np


def q1_func():
    A = np.array(
        [
            [1, .2, .4, .36],
            [.2, 1, .3, .27],
            [.4, .3, 1, .4],
            [.36, .27, .4, 1]
        ]
    )

    h_squared = np.max(A - np.identity(4), axis=1)
    for i in range(100):
        psi_i = 1 - h_squared
        Psi = np.identity(4)
        for j in range(4):
            Psi[j][j] = psi_i[j]
        print(h_squared)
        Lambda_Squared = A - Psi
        w, v = np.linalg.eig(Lambda_Squared)
        eigen_val_1 = w[0]
        eigen_vector_1 = v[0]
        Lambda = np.array([np.sqrt(eigen_val_1) * eigen_vector_1])
        h_squared = np.sum(np.power(Lambda, 2), 0)
        print('----')
        print(f'eigen_val_1: {eigen_val_1}')
        print(f'eigen_vector_1: {eigen_vector_1}')
        print(f'Lambda: {Lambda}')
        print(f'Psi: {Psi}')


def main():
    q1_func()


if __name__ == '__main__':
    main()
