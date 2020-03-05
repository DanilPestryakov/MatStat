import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import factorial

countOfData = [10, 50, 1000]

plt.suptitle('Normal Distribution')
mu, sigma = 0, 1
for i in range(len(countOfData)):
    plt.subplot(1, 3, i + 1)
    N = np.random.normal(mu, sigma, countOfData[i])
    count, bins, ignored = plt.hist(N, 30, normed=True, edgecolor='black')
    plt.plot(bins, 1 / np.sqrt(2 * np.pi) * np.exp(-(bins ** 2) / 2), linewidth=2, color='r')
    plt.title(r'Normal Distribution: $\mu=0$, $\sigma=1$, N=%i' % countOfData[i])
    plt.xlabel('NormalNumbers')
    plt.ylabel('Density')
    plt.subplots_adjust(wspace=0.5)
plt.show()

plt.suptitle('Laplace Distribution')
mu, sigma = 0, math.sqrt(2)
for i in range(len(countOfData)):
    plt.subplot(1, 3, i + 1)
    L = np.random.laplace(mu, sigma, countOfData[i])
    count, bins, ignored = plt.hist(L, 30, normed=True, edgecolor='black')
    plt.plot(bins, 1 / np.sqrt(2) * np.exp(-np.sqrt(2) * np.fabs(bins)), linewidth=2, color='r')
    plt.title(r'Laplace Distribution: $\mu=0$, $\sigma=1.41$, N=%i' % countOfData[i])
    plt.xlabel('LaplaceNumbers')
    plt.ylabel('Density')
    plt.subplots_adjust(wspace=0.5)
plt.show()

plt.suptitle('Poisson Distribution')
base = 10
for i in range(len(countOfData)):
    plt.subplot(1, 3, i + 1)
    P = np.random.poisson(10, countOfData[i])
    count, bins, ignored = plt.hist(P, 30, normed=True, edgecolor='black')
    plt.plot(bins,  np.exp(-10) * np.power(10, bins) / factorial(bins), linewidth=2, color='r')
    plt.title(r'Poisson Distribution: $base=10$, N=%i' % countOfData[i])
    plt.xlabel('PoissonNumbers')
    plt.ylabel('Density')
    plt.subplots_adjust(wspace=0.5)
plt.show()

plt.suptitle('Cauchy Distribution')
for i in range(len(countOfData)):
    plt.subplot(1, 3, i + 1)
    C = np.random.standard_cauchy(countOfData[i])
    count, bins, ignored = plt.hist(C, 30, normed=True, edgecolor='black')
    plt.plot(bins, 1 / (np.pi * (bins ** 2 + 1)), linewidth=2, color='r')
    plt.title(r'Standard Cauchy Distribution: N=%i' % countOfData[i])
    plt.xlabel('CauchyNumbers')
    plt.ylabel('Density')
    plt.subplots_adjust(wspace=0.5)
plt.show()


plt.suptitle('Uniform Distribution')
a = -np.sqrt(3)
b = np.sqrt(3)
for i in range(len(countOfData)):
    plt.subplot(1, 3, i + 1)
    U = np.random.uniform(-np.sqrt(3), np.sqrt(3), countOfData[i])
    count, bins, ignored = plt.hist(U, 30, normed=True, edgecolor='black')
    ar = np.arange(-3., 3., 0.01)
    listU = []
    for elem in ar:
        listU.append(1 / (2 * np.sqrt(3))) if np.fabs(elem) <= np.sqrt(3) else listU.append(0)
    plt.plot(ar, listU, linewidth=2, color='r')
    plt.title(r'Uniform Distribution: $a=-1.73$, $b=1.73$, N=%i' % countOfData[i])
    plt.xlabel('UniformNumbers')
    plt.ylabel('Density')
    plt.subplots_adjust(wspace=0.9)
plt.show()


