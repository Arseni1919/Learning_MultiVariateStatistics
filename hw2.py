import random
# from scipy.optimize import fsolve
# from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations


def create_list_of_points(curr_dict):
    list_of_points = []
    for i in range(len(curr_dict['x'])):
        list_of_points.append((curr_dict['x'][i], curr_dict['y'][i]))
    return list_of_points


def q_1():
    N = 500
    cov = [
        [1, 2],
        [2, 9]
    ]
    costs = [
        [0, 3, 6],
        [4, 0, 2],
        [5, 1, 0]
    ]
    means = [[0, 0], [2, 0], [1, 1]]
    real_values = {i: {'label': f'group {i + 1}', 'x': [], 'y': []} for i in range(len(means))}
    groups = {i: {'label': f'group {i + 1}', 'x': [], 'y': []} for i in range(len(means))}

    for mean_i, mean in enumerate(means):
        x, y = np.random.multivariate_normal(mean, cov, N).T

        # save real groups
        real_values[mean_i]['x'].extend(x)
        real_values[mean_i]['y'].extend(y)

        # predict groups according the rules
        for i in range(len(x)):
            if y[i] > 7 * x[i] + 3 and y[i] > 4.5*x[i] - 4.5:
                # group 1
                groups[0]['x'].append(x[i])
                groups[0]['y'].append(y[i])
            elif y[i] < 11/3*x[i] - 5 and y[i] < 4.5*x[i] - 4.5:
                # group 2
                groups[1]['x'].append(x[i])
                groups[1]['y'].append(y[i])
            else:
                # group 3
                groups[2]['x'].append(x[i])
                groups[2]['y'].append(y[i])

    for i in range(len(means)):
        plt.plot(groups[i]['x'], groups[i]['y'], 'x', label=groups[i]['label'])
    plt.title('Predicted Groups')
    plt.legend()
    plt.show()

    """
    OUTPUT: graph
    """

    # probabilities for wrong group assignment
    expected_cost = 0
    print('=========== Q1 - SEIF BET ===========')
    for real_i, sorted_j in permutations(range(len(means)), 2):
        real_group = create_list_of_points(real_values[real_i])
        assigned_group = create_list_of_points(groups[sorted_j])
        wrong_assignments = len([i for i in real_group if i in assigned_group])
        prop_to_wrong_assignment = wrong_assignments / len(real_group)
        print(f'dist {real_i + 1} -> sorted to group {sorted_j + 1} with prop: {prop_to_wrong_assignment}')

        # add to expected cost according to costs matrix
        expected_cost += prop_to_wrong_assignment * costs[real_i][sorted_j]

    """
    OUTPUT EXAMPLE:
    dist 1 -> sorted to group 2 with prop: 0.028
    dist 1 -> sorted to group 3 with prop: 0.682
    dist 2 -> sorted to group 1 with prop: 0.0
    dist 2 -> sorted to group 3 with prop: 0.204
    dist 3 -> sorted to group 1 with prop: 0.048
    dist 3 -> sorted to group 2 with prop: 0.214
    """

    print('=========== Q1 - SEIF GIMEL ===========')
    print(f'Expected cost: {expected_cost: .2f}')

    """
    OUTPUT EXAMPLE:
    Expected cost:  5.04
    """


def q_2_seif_gimel():
    print('=========== Q2 - SEIF GIMEL ===========')


def main():

    # Q1
    q_1()

    # Q2
    q_2_seif_gimel()


if __name__ == '__main__':
    seed = 12
    to_use_seed = True
    random.seed(seed)
    np.random.seed(seed)

    main()
