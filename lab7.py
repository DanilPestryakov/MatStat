import numpy as np
from tabulate import tabulate
import scipy.stats as stats

distr = np.random.normal(0, 1, size=100)
mu_n = np.mean(distr)
sigma_n = np.std(distr)
print(np.around(mu_n, decimals=2), ' ', np.around(sigma_n, decimals=2))

alpha = 0.05
p = 1 - alpha
k = 8
n = 100

x = np.arange(-1.5, 1.6, 0.5)
arrCount = np.zeros(8)
arrProb = np.zeros(8)
print(x)
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
arrProb[7] = 1 - stats.norm.cdf(x[6])
print("ni = ", arrCount, "sum = ", np.sum(arrCount))
print("pi = ", arrProb, "sum = ", np.sum(arrProb))
print("n*pi = ", n * arrProb, "sum = ", np.sum(n * arrProb))
print("ni - n * pi = ", arrCount - n * arrProb, "sum = ", np.sum(arrCount - n * arrProb))
print("(ni - n * pi) ^ 2 / (n*pi) = ", ((arrCount - n * arrProb) ** 2) / (n * arrProb), "sum = ", np.sum(((arrCount - n * arrProb) ** 2) / (n * arrProb)))



