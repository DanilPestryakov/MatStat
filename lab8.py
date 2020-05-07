import numpy as np
import scipy.stats as stats

gamma_confidence = 0.95
alfa = 1 - gamma_confidence


def m_confidence(distr):
    mean = np.mean(distr)
    std = np.std(distr)
    n = len(distr)
    interval = std * stats.t.ppf((1 - alfa / 2), n - 1) / (n - 1) ** 0.5
    return np.around(mean - interval, decimals=2), np.around(mean + interval, decimals=2)


def var_confidence(distr):
    std = np.std(distr)
    n = len(distr)
    left_b = std * (n / stats.chi2.ppf((1 - alfa / 2), n - 1)) ** 0.5
    right_b = std * (n / stats.chi2.ppf((alfa / 2), n - 1)) ** 0.5
    return np.around(left_b, decimals=2), np.around(right_b, decimals=2)


def m_confidence_asimpt(distr):
    mean = np.mean(distr)
    std = np.std(distr)
    n = len(distr)
    u = stats.norm.ppf(1 - alfa / 2)
    interval = std * u / (n ** 0.5)
    return np.around(mean - interval, decimals=2), np.around(mean + interval, decimals=2)


def var_confidence_asimpt(distr):
    std = np.std(distr)
    n = len(distr)
    m_4 = stats.moment(distr, 4)
    e_ = m_4 / std**4 - 3
    u = stats.norm.ppf(1 - alfa / 2)
    U = u * (((e_ + 2) / n) ** 0.5)
    left_b = std * (1 + 0.5 * U) ** (-0.5)
    right_b = std * (1 - 0.5 * U) ** (-0.5)
    return np.around(left_b, decimals=2), np.around(right_b, decimals=2)


size = [20, 100]
for s in size:
    distr = np.random.normal(0, 1, size=s)
    print('size = ' + str(s))
    print('mean', m_confidence(distr))
    print('variance', var_confidence(distr))
    print('asimpt_mean', m_confidence_asimpt(distr))
    print('asimpt_variance', var_confidence_asimpt(distr))