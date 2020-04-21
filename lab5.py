import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import scipy.stats as stats
from matplotlib.patches import Ellipse
from tabulate import tabulate
import statistics

sizes = [20, 60, 100]
rcor = [0, 0.5, 0.9]

def comp_quadrant(x, y):
    size = len(x)
    med_x = np.median(x)
    med_y = np.median(y)
    x_new = np.empty(size, dtype=float)
    x_new.fill(med_x)
    x_new = x - x_new
    y_new = np.empty(size, dtype=float)
    y_new.fill(med_y)
    y_new = y - y_new
    n = [0, 0, 0, 0]
    for i in range(size):
        if x_new[i] >= 0 and y_new[i] >= 0:
            n[0] += 1
        if x_new[i] < 0 and y_new[i] > 0:
            n[1] += 1
        if x_new[i] < 0 and y_new[i] < 0:
            n[2] += 1
        if x_new[i] > 0 and y_new[i] < 0:
            n[3] += 1
    return ((n[0] + n[2]) - (n[1] + n[3])) / size

def correlation_coef(size, rcor, repeat):
    math_exp = [0, 0]
    div = [[1.0, rcor], [rcor, 1.0]]
    quadrant_coef = np.empty(repeat, dtype=float)
    pearson_coef = np.empty(repeat, dtype=float)
    spearman_coef = np.empty(repeat, dtype=float)
    for i in range(repeat):
        rv = stats.multivariate_normal.rvs(math_exp, div, size=size)
        x = rv[:, 0]
        y = rv[:, 1]
        quadrant_coef[i] = comp_quadrant(x, y)
        pearson_coef[i], t = stats.pearsonr(x, y)
        spearman_coef[i], t = stats.spearmanr(x, y)
    return quadrant_coef, pearson_coef, spearman_coef

def create_table(quadrant_coef, pearson_coef, spearman_coef, rcor, size, repeat):
    rows = []
    headers = []
    if rcor != -1:
        rows.append(["rho = " + str(rcor), 'r', 'r_{S}', 'r_{Q}'])
    else:
        rows.append(["n = " + str(size), 'r', 'r_{S}', 'r_{Q}'])
    q = np.mean(quadrant_coef)
    p = np.mean(pearson_coef)
    s = np.mean(spearman_coef)
    rows.append(['E(z)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])
    quadrant_coef_quad = np.empty(repeat, dtype=float)
    pearson_coef_quad = np.empty(repeat, dtype=float)
    spearman_coef_quad = np.empty(repeat, dtype=float)
    for i in range(repeat):
        quadrant_coef_quad[i] = quadrant_coef[i] * quadrant_coef[i]
        pearson_coef_quad[i] = pearson_coef[i] * pearson_coef[i]
        spearman_coef_quad[i] = spearman_coef[i] * spearman_coef[i]
    q = np.mean(quadrant_coef_quad)
    p = np.mean(pearson_coef_quad)
    s = np.mean(spearman_coef_quad)
    rows.append(['E(z^2)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])
    q = statistics.variance(quadrant_coef)
    p = statistics.variance(pearson_coef)
    s = statistics.variance(spearman_coef)
    rows.append(['D(z)', np.around(p, decimals=3), np.around(s, decimals=3), np.around(q, decimals=3)])
    print(tabulate(rows, headers, tablefmt="latex"))
    print('\n')

def set_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



def plot_ellipse(size):
    mean = [0, 0]
    fig, ax = plt.subplots(1, 3)
    fig.suptitle("n = " + str(size))
    titles = [r'$ \rho = 0$', r'$\rho = 0.5 $', r'$ \rho = 0.9$']
    num = 0
    for r in rcor:
        cov = [[1.0, r], [r, 1.0]]
        rv = stats.multivariate_normal.rvs(mean, cov, size=size)
        x = rv[:, 0]
        y = rv[:, 1]
        ax[num].scatter(x, y, s=3)
        set_ellipse(x, y, ax[num], edgecolor='navy')
        ax[num].scatter(np.mean(x), np.mean(y), c='aqua', s=3)
        ax[num].set_title(titles[num])
        num += 1
    #plt.savefig("n" + str(size) + ".png", format='png')
    plt.show()

repeat = 1000
for i in sizes:
    for j in rcor:
        quadrant_coef, pearson_coef, spearman_coef = correlation_coef(i, j, repeat)
        create_table(quadrant_coef, pearson_coef, spearman_coef, j, i, repeat)
    pearson_coef = np.empty(repeat, dtype=float)
    spearman_coef = np.empty(repeat, dtype=float)
    quadrant_coef = np.empty(repeat, dtype=float)
    for k in range(repeat):
        rv = []
        for l in range(2):
            x = 0.9 * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], i) + 0.1 * stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], i)
            rv += list(x)
        rv = np.array(rv)
        x = rv[:, 0]
        y = rv[:, 1]
        pearson_coef[k], t = stats.pearsonr(x, y)
        spearman_coef[k], t = stats.spearmanr(x, y)
        quadrant_coef[k] = comp_quadrant(x, y)
    create_table(quadrant_coef, pearson_coef, spearman_coef, -1, i, repeat)
    plot_ellipse(i)