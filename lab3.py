import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy.special as sc

X1 = np.random.normal(0, 1, 20)
X2 = np.random.normal(0, 1, 100)
line_props = dict(color="black", alpha=0.3, linestyle="dashdot")
bbox_props = dict(color="b", alpha=0.9)
flier_props = dict(marker="o", markersize=4)
plt.boxplot((X1, X2), whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props, labels=["n = 20", "n = 100"])
plt.ylabel("X")
plt.title("Normal")
plt.show()

mu, sigma = 0, np.sqrt(2)
X1 = np.random.laplace(mu, sigma, 20)
X2 = np.random.laplace(mu, sigma, 100)
line_props = dict(color="black", alpha=0.3, linestyle="dashdot")
bbox_props = dict(color="b", alpha=0.9)
flier_props = dict(marker="o", markersize=4)
plt.boxplot((X1, X2), whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props, labels=["n = 20", "n = 100"])
plt.ylabel("X")
plt.title("Laplace")
plt.show()

X1 = np.random.poisson(10, 20)
X2 = np.random.poisson(10, 100)
line_props = dict(color="black", alpha=0.3, linestyle="dashdot")
bbox_props = dict(color="b", alpha=0.9)
flier_props = dict(marker="o", markersize=4)
plt.boxplot((X1, X2), whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props, labels=["n = 20", "n = 100"])
plt.ylabel("X")
plt.title("Poisson")
plt.show()

X1 = np.random.standard_cauchy(20)
X2 = np.random.standard_cauchy(100)
line_props = dict(color="black", alpha=0.3, linestyle="dashdot")
bbox_props = dict(color="b", alpha=0.9)
flier_props = dict(marker="o", markersize=4)
plt.boxplot((X1, X2), whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props, labels=["n = 20", "n = 100"])
plt.ylabel("X")
plt.title("Cauchy")
plt.show()

X1 = np.random.uniform(-np.sqrt(3), np.sqrt(3), 20)
X2 = np.random.uniform(-np.sqrt(3), np.sqrt(3), 100)
line_props = dict(color="black", alpha=0.3, linestyle="dashdot")
bbox_props = dict(color="b", alpha=0.9)
flier_props = dict(marker="o", markersize=4)
plt.boxplot((X1, X2), whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props, labels=["n = 20", "n = 100"])
plt.ylabel("X")
plt.title("Uniform")
plt.show()


countOfIter = 1000
row = []
headers = ["distribution name", "proportion of ejections"]

count = [0, 0]
for i in range(countOfIter):
    X1 = np.random.normal(0, 1, 20)
    X2 = np.random.normal(0, 1, 100)

    prop_20 = []
    prop_100 = []

    prop_20.append(np.quantile(X1, 0.25) - 1.5 * (np.quantile(X1, 0.75) - np.quantile(X1, 0.25)))
    prop_20.append(np.quantile(X1, 0.75) + 1.5 * (np.quantile(X1, 0.75) - np.quantile(X1, 0.25)))

    prop_100.append(np.quantile(X2, 0.25) - 1.5 * (np.quantile(X2, 0.75) - np.quantile(X2, 0.25)))
    prop_100.append(np.quantile(X2, 0.75) + 1.5 * (np.quantile(X2, 0.75) - np.quantile(X2, 0.25)))

    for k in range(0, 20):
        if X1[k] > prop_20[1] or X1[k] < prop_20[0]:
            count[0] = count[0] + 1

    for k in range(0, 100):
        if X2[k] > prop_100[1] or X2[k] < prop_100[0]:
            count[1] = count[1] + 1
count[0] /= 1000
count[1] /= 1000
row.append(["normal n = 20", np.around(count[0] / 20, decimals=3)])
row.append(["normal n = 100", np.around(count[1] / 100, decimals=3)])


count = [0, 0]
for i in range(countOfIter):
    X1 = np.random.standard_cauchy(20)
    X2 = np.random.standard_cauchy(100)

    prop_20 = []
    prop_100 = []

    prop_20.append(np.quantile(X1, 0.25) - 1.5 * (np.quantile(X1, 0.75) - np.quantile(X1, 0.25)))
    prop_20.append(np.quantile(X1, 0.75) + 1.5 * (np.quantile(X1, 0.75) - np.quantile(X1, 0.25)))

    prop_100.append(np.quantile(X2, 0.25) - 1.5 * (np.quantile(X2, 0.75) - np.quantile(X2, 0.25)))
    prop_100.append(np.quantile(X2, 0.75) + 1.5 * (np.quantile(X2, 0.75) - np.quantile(X2, 0.25)))

    for k in range(0, 20):
        if X1[k] > prop_20[1] or X1[k] < prop_20[0]:
            count[0] = count[0] + 1

    for k in range(0, 100):
        if X2[k] > prop_100[1] or X2[k] < prop_100[0]:
            count[1] = count[1] + 1
count[0] /= 1000
count[1] /= 1000
row.append(["cauchy n = 20", np.around(count[0] / 20, decimals=3)])
row.append(["cauchy n = 100", np.around(count[1] / 100, decimals=3)])


count = [0, 0]
mu, sigma = 0, np.sqrt(2)
for i in range(countOfIter):
    X1 = np.random.laplace(mu, sigma, 20)
    X2 = np.random.laplace(mu, sigma, 100)

    prop_20 = []
    prop_100 = []

    prop_20.append(np.quantile(X1, 0.25) - 1.5 * (np.quantile(X1, 0.75) - np.quantile(X1, 0.25)))
    prop_20.append(np.quantile(X1, 0.75) + 1.5 * (np.quantile(X1, 0.75) - np.quantile(X1, 0.25)))

    prop_100.append(np.quantile(X2, 0.25) - 1.5 * (np.quantile(X2, 0.75) - np.quantile(X2, 0.25)))
    prop_100.append(np.quantile(X2, 0.75) + 1.5 * (np.quantile(X2, 0.75) - np.quantile(X2, 0.25)))

    for k in range(0, 20):
        if X1[k] > prop_20[1] or X1[k] < prop_20[0]:
            count[0] = count[0] + 1

    for k in range(0, 100):
        if X2[k] > prop_100[1] or X2[k] < prop_100[0]:
            count[1] = count[1] + 1
count[0] /= 1000
count[1] /= 1000
row.append(["laplace n = 20", np.around(count[0] / 20, decimals=3)])
row.append(["laplace n = 100", np.around(count[1] / 100, decimals=3)])


count = [0, 0]
for i in range(countOfIter):
    X1 = np.random.poisson(10, 20)
    X2 = np.random.poisson(10, 100)

    prop_20 = []
    prop_100 = []

    prop_20.append(np.quantile(X1, 0.25) - 1.5 * (np.quantile(X1, 0.75) - np.quantile(X1, 0.25)))
    prop_20.append(np.quantile(X1, 0.75) + 1.5 * (np.quantile(X1, 0.75) - np.quantile(X1, 0.25)))

    prop_100.append(np.quantile(X2, 0.25) - 1.5 * (np.quantile(X2, 0.75) - np.quantile(X2, 0.25)))
    prop_100.append(np.quantile(X2, 0.75) + 1.5 * (np.quantile(X2, 0.75) - np.quantile(X2, 0.25)))

    for k in range(0, 20):
        if X1[k] > prop_20[1] or X1[k] < prop_20[0]:
            count[0] = count[0] + 1

    for k in range(0, 100):
        if X2[k] > prop_100[1] or X2[k] < prop_100[0]:
            count[1] = count[1] + 1
count[0] /= 1000
count[1] /= 1000
row.append(["pois n = 20", np.around(count[0] / 20, decimals=3)])
row.append(["pois n = 100", np.around(count[1] / 100, decimals=3)])


count = [0, 0]
for i in range(countOfIter):
    X1 = np.random.uniform(-np.sqrt(3), np.sqrt(3), 20)
    X2 = np.random.uniform(-np.sqrt(3), np.sqrt(3), 100)

    prop_20 = []
    prop_100 = []

    prop_20.append(np.quantile(X1, 0.25) - 1.5 * (np.quantile(X1, 0.75) - np.quantile(X1, 0.25)))
    prop_20.append(np.quantile(X1, 0.75) + 1.5 * (np.quantile(X1, 0.75) - np.quantile(X1, 0.25)))

    prop_100.append(np.quantile(X2, 0.25) - 1.5 * (np.quantile(X2, 0.75) - np.quantile(X2, 0.25)))
    prop_100.append(np.quantile(X2, 0.75) + 1.5 * (np.quantile(X2, 0.75) - np.quantile(X2, 0.25)))

    for k in range(0, 20):
        if X1[k] > prop_20[1] or X1[k] < prop_20[0]:
            count[0] = count[0] + 1

    for k in range(0, 100):
        if X2[k] > prop_100[1] or X2[k] < prop_100[0]:
            count[1] = count[1] + 1
count[0] /= 1000
count[1] /= 1000
row.append(["unif n = 20", np.around(count[0] / 20, decimals=3)])
row.append(["unif n = 100", np.around(count[1] / 100, decimals=3)])

print(tabulate(row, headers, tablefmt="latex"))
print("\n")