import numpy as np
from tabulate import tabulate
import scipy.stats as stats

def check_theory(mode, sizeProb):
    if mode == "Laplace":
        distr = np.random.laplace(0, 1, sizeProb)
        k = 6
        x = np.arange(-1.5, 1.6, 0.75)
    if mode == "Norm":
        distr = np.random.normal(0, 1, sizeProb)
        k = 8
        x = np.arange(-1.5, 1.6, 0.5)
    mu_n = np.mean(distr)
    sigma_n = np.std(distr)
    print(np.around(mu_n, decimals=2), ' ', np.around(sigma_n, decimals=2))

    alpha = 0.05
    p = 1 - alpha
    n = sizeProb

    arrCount = np.zeros(k)
    arrProb = np.zeros(k)
    print(x)
    if mode == "Laplace":
        for i in range(len(distr)):
            if distr[i] <= x[0]:
                arrCount[0] += 1
            elif distr[i] > x[k - 2]:
                arrCount[k - 1] += 1
            else:
                for j in range(k - 2):
                    if (distr[i] > x[j]) & (distr[i] <= x[j + 1]):
                        arrCount[j + 1] += 1
                        break
        for i in range(k - 2):
            arrProb[i + 1] = (stats.laplace.cdf(x[i + 1]) - stats.laplace.cdf(x[i]))
        arrProb[0] = stats.laplace.cdf(x[0])
        arrProb[k - 1] = 1 - stats.laplace.cdf(x[k - 2])
    if mode == "Norm":
        for i in range(len(distr)):
            if distr[i] <= x[0]:
                arrCount[0] += 1
            elif distr[i] > x[k - 2]:
                arrCount[k - 1] += 1
            else:
                for j in range(k - 2):
                    if (distr[i] > x[j]) & (distr[i] <= x[j + 1]):
                        arrCount[j + 1] += 1
                        break
        for i in range(k - 2):
            arrProb[i + 1] = (stats.norm.cdf(x[i + 1]) - stats.norm.cdf(x[i]))
        arrProb[0] = stats.norm.cdf(x[0])
        arrProb[k - 1] = 1 - stats.norm.cdf(x[k - 2])
    print("ni = ", arrCount, "sum = ", np.sum(arrCount))
    print("pi = ", arrProb, "sum = ", np.sum(arrProb))
    print("n*pi = ", n * arrProb, "sum = ", np.sum(n * arrProb))
    print("ni - n * pi = ", arrCount - n * arrProb, "sum = ", np.sum(arrCount - n * arrProb))
    print("(ni - n * pi) ^ 2 / (n*pi) = ", ((arrCount - n * arrProb) ** 2) / (n * arrProb), "sum = ", np.sum(((arrCount - n * arrProb) ** 2) / (n * arrProb)))


check_theory("Laplace", 30)



