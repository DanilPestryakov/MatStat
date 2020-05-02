import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt


def mnk_coefficients(x, y):
    beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x * x) - np.mean(x) ** 2)
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    return beta_0, beta_1

def quart(array, p):
    new_array = np.sort(array)
    k = len(array) * p
    if k.is_integer():
        return new_array[int(k)]
    else:
        return new_array[int(k) + 1]

def cor_coef(x, y):
    medX = np.median(x)
    medY = np.median(y)
    sum = 0
    for i in range(len(x)):
        if (x[i] - medX) * (y[i] - medY) > 0:
            sum += 1
        if (x[i] - medX) * (y[i] - medY) < 0:
            sum -= 1
    sum /= len(x)
    return sum

def mna_coefficients(x, y):
    beta_1 = cor_coef(x, y) * (quart(y, 0.75) - quart(y, 0.25)) / (quart(x, 0.75) - quart(x, 0.25))
    beta_0 = np.median(y) - beta_1 * np.median(x)
    return beta_0, beta_1


def plot_regr(x, y, type):
    Kbeta_0, Kbeta_1 = mnk_coefficients(x, y)
    print('MNK')
    print('beta_0 = ' + str(np.around(Kbeta_0, decimals=2)))
    print('beta_1 = ' + str(np.around(Kbeta_1, decimals=2)))
    Abeta_0, Abeta_1 = mna_coefficients(x, y)
    print('MNA')
    print('beta_0 = ' + str(np.around(Abeta_0, decimals=2)))
    print('beta_1 = ' + str(np.around(Abeta_1, decimals=2)))
    plt.scatter(x[1:-2], y[1:-2], label='Выборка', edgecolor='navy')
    plt.plot(x, x * (2 * np.ones(len(x))) + 2 * np.ones(len(x)), label='Модель', color='blue')
    plt.plot(x, x * (Kbeta_1 * np.ones(len(x))) + Kbeta_0 * np.ones(len(x)), label='МHK', color='deepskyblue')
    plt.plot(x, x * (Abeta_1 * np.ones(len(x))) + Abeta_0 * np.ones(len(x)), label='МHM', color='indigo')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim([-1.8, 2])
    plt.legend()
    plt.title(type)
    plt.savefig(type + '.png', format='png')
    plt.show()


if __name__ == '__main__':
    x = np.arange(-1.8, 2, 0.2)
    y = 2 * x + 2 * np.ones(len(x)) + np.random.normal(0, 1, size=len(x))
    types = ['Без возмущений', 'С возмущениями']
    plot_regr(x, y, types[0])
    y[0] += 10
    y[-1] -= 10
    plot_regr(x, y, types[1])