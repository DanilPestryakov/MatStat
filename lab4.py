import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.distributions.empirical_distribution import ECDF
import seaborn as sns

array_20 = stats.norm.rvs(0, 1, 20)
array_60 = stats.norm.rvs(0, 1, 60)
array_100 = stats.norm.rvs(0, 1, 100)
arrays = [array_20, array_60, array_100]
quan_of_numbers = [20, 60, 100]
j = 1
array_global = np.arange(-4, 4, 0.01)
for array in arrays:
    for i in array:
        if i < -4 or i > 4:
            array = np.delete(array, list(array).index(i))
    plt.subplot(1, 3, j)
    plt.title('Normal, n = ' + str(quan_of_numbers[j - 1]))
    plt.plot(array_global, stats.norm.cdf(array_global), color='blue', linewidth=0.8)
    array_ex = np.linspace(-4, 4)
    ecdf = ECDF(array)
    y = ecdf(array_ex)
    plt.step(array_ex, y, color='black')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.subplots_adjust(wspace=0.5)
    j += 1
plt.show()

k = 1
for array in arrays:
    for i in array:
        if i < -4 or i > 4:
            array = np.delete(array, list(array).index(i))
    titles = [r'$h = \frac{h_n}{2}$', r'$h = h_n$', r'$h = 2 * h_n$']
    l = 0
    fig, ax = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.5)
    for bandwidth in [0.5, 1, 2]:
        kde = stats.gaussian_kde(array, bw_method='silverman')
        h_n = kde.factor
        fig.suptitle('Normal, n = ' + str(quan_of_numbers[k - 1]))
        ax[l].plot(array_global, stats.norm.pdf(array_global, 0, 1), color='blue', alpha=0.5, label='density')
        ax[l].set_title(titles[l])
        sns.kdeplot(array, ax=ax[l], bw=h_n*bandwidth, label='kde')
        ax[l].set_xlabel('x')
        ax[l].set_ylabel('f(x)')
        ax[l].set_ylim([0, 1])
        ax[l].set_xlim([-4, 4])
        ax[l].legend()
        l += 1
    plt.show()
    k += 1


array_20 = np.random.standard_cauchy(20)
array_60 = np.random.standard_cauchy(60)
array_100 = np.random.standard_cauchy(100)
arrays = [array_20, array_60, array_100]
j = 1
array_global = np.arange(-4, 4, 0.01)
for array in arrays:
    for i in array:
        if i < -4 or i > 4:
            array = np.delete(array, list(array).index(i))
    plt.subplot(1, 3, j)
    plt.title('Cauchy, n = ' + str(quan_of_numbers[j - 1]))
    plt.plot(array_global, stats.cauchy.cdf(array_global), color='blue', linewidth=0.8)
    array_ex = np.linspace(-4, 4)
    ecdf = ECDF(array)
    y = ecdf(array_ex)
    plt.step(array_ex, y, color='black')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.subplots_adjust(wspace=0.5)
    j += 1
plt.show()


k = 1
for array in arrays:
    for i in array:
        if i < -4 or i > 4:
            array = np.delete(array, list(array).index(i))
    titles = [r'$h = \frac{h_n}{2}$', r'$h = h_n$', r'$h = 2 * h_n$']
    l = 0
    fig, ax = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.5)
    for bandwidth in [0.5, 1, 2]:
        kde = stats.gaussian_kde(array, bw_method='silverman')
        h_n = kde.factor
        fig.suptitle('Cauchy, n = ' + str(quan_of_numbers[k - 1]))
        ax[l].plot(array_global, stats.cauchy.pdf(array_global), color='blue', alpha=0.5, label='density')
        ax[l].set_title(titles[l])
        sns.kdeplot(array, ax=ax[l], bw=h_n*bandwidth, label='kde')
        ax[l].set_xlabel('x')
        ax[l].set_ylabel('f(x)')
        ax[l].set_ylim([0, 1])
        ax[l].set_xlim([-4, 4])
        ax[l].legend()
        l += 1
    plt.show()
    k += 1


array_20 = stats.laplace.rvs(0, 1 / (2 ** 0.5), 20)
array_60 = stats.laplace.rvs(0, 1 / (2 ** 0.5), 60)
array_100 = stats.laplace.rvs(0, 1 / (2 ** 0.5), 100)
arrays = [array_20, array_60, array_100]
j = 1
array_global = np.arange(-4, 4, 0.01)
for array in arrays:
    for i in array:
        if i < -4 or i > 4:
            array = np.delete(array, list(array).index(i))
    plt.subplot(1, 3, j)
    plt.title('Laplace, n = ' + str(quan_of_numbers[j - 1]))
    plt.plot(array_global, stats.laplace.cdf(array_global), color='blue', linewidth=0.8)
    array_ex = np.linspace(-4, 4)
    ecdf = ECDF(array)
    y = ecdf(array_ex)
    plt.step(array_ex, y, color='black')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.subplots_adjust(wspace=0.5)
    j += 1
plt.show()


k = 1
for array in arrays:
    for i in array:
        if i < -4 or i > 4:
            array = np.delete(array, list(array).index(i))
    titles = [r'$h = \frac{h_n}{2}$', r'$h = h_n$', r'$h = 2 * h_n$']
    l = 0
    fig, ax = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.5)
    for bandwidth in [0.5, 1, 2]:
        kde = stats.gaussian_kde(array, bw_method='silverman')
        h_n = kde.factor
        fig.suptitle('Laplace, n = ' + str(quan_of_numbers[k - 1]))
        ax[l].plot(array_global, stats.laplace.pdf(array_global, 0, 1 / 2 ** 0.5), color='blue', alpha=0.5, label='density')
        ax[l].set_title(titles[l])
        sns.kdeplot(array, ax=ax[l], bw=h_n*bandwidth, label='kde')
        ax[l].set_xlabel('x')
        ax[l].set_ylabel('f(x)')
        ax[l].set_ylim([0, 1])
        ax[l].set_xlim([-4, 4])
        ax[l].legend()
        l += 1
    plt.show()
    k += 1

array_20 = np.random.poisson(10, 20)
array_60 = np.random.poisson(10, 60)
array_100 = np.random.poisson(10, 100)
arrays = [array_20, array_60, array_100]
j = 1
array_global = np.arange(6, 14, 1)
for array in arrays:
    for i in array:
        if i < 6 or i > 14:
            array = np.delete(array, list(array).index(i))
    plt.subplot(1, 3, j)
    plt.title('Poisson, n = ' + str(quan_of_numbers[j - 1]))
    plt.plot(array_global, stats.poisson.cdf(array_global, 10), color='blue', linewidth=0.8)
    array_ex = np.linspace(6, 14)
    ecdf = ECDF(array)
    y = ecdf(array_ex)
    plt.step(array_ex, y, color='black')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.subplots_adjust(wspace=0.5)
    j += 1
plt.show()


k = 1
for array in arrays:
    for i in array:
        if i < 6 or i > 14:
            array = np.delete(array, list(array).index(i))
    titles = [r'$h = \frac{h_n}{2}$', r'$h = h_n$', r'$h = 2 * h_n$']
    l = 0
    fig, ax = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.5)
    for bandwidth in [0.5, 1, 2]:
        kde = stats.gaussian_kde(array, bw_method='silverman')
        h_n = kde.factor
        fig.suptitle('Poisson, n = ' + str(quan_of_numbers[k - 1]))
        ax[l].plot(array_global, stats.poisson.pmf(10, array_global), color='blue', alpha=0.5, label='density')
        ax[l].set_title(titles[l])
        sns.kdeplot(array, ax=ax[l], bw=h_n*bandwidth, label='kde')
        ax[l].set_xlabel('x')
        ax[l].set_ylabel('f(x)')
        ax[l].set_ylim([0, 1])
        ax[l].set_xlim([6, 14])
        ax[l].legend()
        l += 1
    plt.show()
    k += 1

mu = float(3 ** 0.5)
array_20 = stats.uniform.rvs(-mu, 2 * mu, 20)
array_60 = stats.uniform.rvs(-mu, 2 * mu, 60)
array_100 = stats.uniform.rvs(-mu, 2 * mu, 100)
arrays = [array_20, array_60, array_100]
j = 1
array_global = np.arange(-4, 4, 0.01)
for array in arrays:
    for i in array:
        if i < -4 or i > 4:
            array = np.delete(array, list(array).index(i))
    plt.subplot(1, 3, j)
    plt.title('Unif, n = ' + str(quan_of_numbers[j - 1]))
    plt.plot(array_global, stats.uniform.cdf(array_global), color='blue', linewidth=0.8)
    array_ex = np.linspace(-4, 4)
    ecdf = ECDF(array)
    y = ecdf(array_ex)
    plt.step(array_ex, y, color='black')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.subplots_adjust(wspace=0.5)
    j += 1
plt.show()


k = 1
for array in arrays:
    for i in array:
        if i < -4 or i > 4:
            array = np.delete(array, list(array).index(i))
    titles = [r'$h = \frac{h_n}{2}$', r'$h = h_n$', r'$h = 2 * h_n$']
    l = 0
    fig, ax = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.5)
    for bandwidth in [0.5, 1, 2]:
        kde = stats.gaussian_kde(array, bw_method='silverman')
        h_n = kde.factor
        fig.suptitle('Unif, n = ' + str(quan_of_numbers[k - 1]))
        ax[l].plot(array_global, stats.uniform.pdf(array_global, -mu, 2 * mu), color='blue', alpha=0.5, label='density')
        ax[l].set_title(titles[l])
        sns.kdeplot(array, ax=ax[l], bw=h_n*bandwidth, label='kde')
        ax[l].set_xlabel('x')
        ax[l].set_ylabel('f(x)')
        ax[l].set_ylim([0, 1])
        ax[l].set_xlim([-4, 4])
        ax[l].legend()
        l += 1
    plt.show()
    k += 1
