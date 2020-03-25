import numpy as np
import math
from tabulate import tabulate

def SredArith(values):
    res = 0.0
    for i in range(len(values)):
        res += (values[i] / len(values))
    return res

def Disp(values):
    sred = SredArith(values)
    res = 0.0
    listRes = []
    for i in range(len(values)):
        res += ((values[i] - sred) * (values[i] - sred) / len(values))
    listRes.append(sred)
    listRes.append(res)
    return listRes

countOfData = [10, 100, 1000]

countOfIter = 1000

countOfValues = 5

Nvalues = []
Lvalues = []
Pvalues = []
Cvalues = []
Uvalues = []


for j in range(len(countOfData)):
    sredN = []
    sredL = []
    sredP = []
    sredC = []
    sredU = []
    medN = []
    medL = []
    medP = []
    medC = []
    medU = []
    RN = []
    RL = []
    RP = []
    RC = []
    RU = []
    QN = []
    QL = []
    QP = []
    QC = []
    QU = []
    TrN = []
    TrL = []
    TrP = []
    TrC = []
    TrU = []
    for i in range(countOfIter):
        N = np.random.normal(0, 1, countOfData[j])
        L = np.random.laplace(0, math.sqrt(2), countOfData[j])
        P = np.random.poisson(10, countOfData[j])
        C = np.random.standard_cauchy(countOfData[j])
        U = np.random.uniform(-np.sqrt(3), np.sqrt(3), countOfData[j])
        N.sort()
        L.sort()
        P.sort()
        C.sort()
        U.sort()
        sumN = 0
        sumL = 0
        sumP = 0
        sumC = 0
        sumU = 0
        for k in range(countOfData[j]):
            sumN += (N[k] / countOfData[j])
            sumL += (L[k] / countOfData[j])
            sumP += (P[k] / countOfData[j])
            sumC += (C[k] / countOfData[j])
            sumU += (U[k] / countOfData[j])
        mdN = (N[countOfData[j] // 2] + N[countOfData[j] // 2 - 1]) / 2
        mdL = (L[countOfData[j] // 2] + L[countOfData[j] // 2 - 1]) / 2
        mdP = (P[countOfData[j] // 2] + P[countOfData[j] // 2 - 1]) / 2
        mdC = (C[countOfData[j] // 2] + C[countOfData[j] // 2 - 1]) / 2
        mdU = (U[countOfData[j] // 2] + U[countOfData[j] // 2 - 1]) / 2
        sredN.append(sumN)
        sredL.append(sumL)
        sredP.append(sumP)
        sredC.append(sumC)
        sredU.append(sumU)
        medN.append(mdN)
        medL.append(mdL)
        medP.append(mdP)
        medC.append(mdC)
        medU.append(mdU)
        RN.append(((N[countOfData[j] - 1] + N[0]) / 2))
        RL.append(((L[countOfData[j] - 1] + N[0]) / 2))
        RP.append(((P[countOfData[j] - 1] + N[0]) / 2))
        RC.append(((C[countOfData[j] - 1] + N[0]) / 2))
        RU.append(((U[countOfData[j] - 1] + N[0]) / 2))
        if (countOfData[j] / 4).is_integer():
            QN.append(((N[countOfData[j] // 4 - 1] + N[3 * countOfData[j] // 4 - 1]) / 2))
            QL.append(((L[countOfData[j] // 4 - 1] + L[3 * countOfData[j] // 4 - 1]) / 2))
            QP.append(((P[countOfData[j] // 4 - 1] + P[3 * countOfData[j] // 4 - 1]) / 2))
            QC.append(((C[countOfData[j] // 4 - 1] + C[3 * countOfData[j] // 4 - 1]) / 2))
            QU.append(((U[countOfData[j] // 4 - 1] + U[3 * countOfData[j] // 4 - 1]) / 2))
        else:
            QN.append(((N[countOfData[j] // 4] + N[3 * countOfData[j] // 4]) / 2))
            QL.append(((L[countOfData[j] // 4] + L[3 * countOfData[j] // 4]) / 2))
            QP.append(((P[countOfData[j] // 4] + P[3 * countOfData[j] // 4]) / 2))
            QC.append(((C[countOfData[j] // 4] + C[3 * countOfData[j] // 4]) / 2))
            QU.append(((U[countOfData[j] // 4] + U[3 * countOfData[j] // 4]) / 2))
        r = countOfIter // 4
        TN = 0
        TL = 0
        TP = 0
        TC = 0
        TU = 0
        for k in range(r, countOfData[j] - r):
            TN += (N[k] / (countOfData[j] - 2 * r))
            TL += (L[k] / (countOfData[j] - 2 * r))
            TP += (P[k] / (countOfData[j] - 2 * r))
            TC += (C[k] / (countOfData[j] - 2 * r))
            TU += (U[k] / (countOfData[j] - 2 * r))
        TrN.append(TN)
        TrL.append(TL)
        TrP.append(TP)
        TrC.append(TC)
        TrU.append(TU)
    Nvalues.append(sredN)
    Nvalues.append(medN)
    Nvalues.append(RN)
    Nvalues.append(QN)
    Nvalues.append(TrN)
    Lvalues.append(sredL)
    Lvalues.append(medL)
    Lvalues.append(RL)
    Lvalues.append(QL)
    Lvalues.append(TrL)
    Pvalues.append(sredP)
    Pvalues.append(medP)
    Pvalues.append(RP)
    Pvalues.append(QP)
    Pvalues.append(TrP)
    Cvalues.append(sredC)
    Cvalues.append(medC)
    Cvalues.append(RC)
    Cvalues.append(QC)
    Cvalues.append(TrC)
    Uvalues.append(sredU)
    Uvalues.append(medU)
    Uvalues.append(RU)
    Uvalues.append(QU)
    Uvalues.append(TrU)

table = []
for i in range(len(countOfData)):
    strS = "normal n = " + str(countOfData[i])
    table.append([strS, "", "", "", "", ""])
    table.append(["", "x", "med x", "zR", "zQ", "zTr"])
    res1 = ["E(z)"]
    res2 = ["D(z)"]
    for j in range(countOfValues):
        res = Disp(Nvalues[i * countOfValues + j])
        res1.append(res[0])
        res2.append(res[1])
    table.append(res1)
    table.append(res2)
    table.append(["", "", "", "", "", ""])
print(tabulate(table))
print(tabulate(table, tablefmt="latex"))


table = []
for i in range(len(countOfData)):
    strS = "cauchy n = " + str(countOfData[i])
    table.append([strS, "", "", "", "", ""])
    table.append(["", "x", "med x", "zR", "zQ", "zTr"])
    res1 = ["E(z)"]
    res2 = ["D(z)"]
    for j in range(countOfValues):
        res = Disp(Cvalues[i * countOfValues + j])
        res1.append(res[0])
        res2.append(res[1])
    table.append(res1)
    table.append(res2)
    table.append(["", "", "", "", "", ""])
print(tabulate(table))
print(tabulate(table, tablefmt="latex"))


table = []
for i in range(len(countOfData)):
    strS = "laplace n = " + str(countOfData[i])
    table.append([strS, "", "", "", "", ""])
    table.append(["", "x", "med x", "zR", "zQ", "zTr"])
    res1 = ["E(z)"]
    res2 = ["D(z)"]
    for j in range(countOfValues):
        res = Disp(Lvalues[i * countOfValues + j])
        res1.append(res[0])
        res2.append(res[1])
    table.append(res1)
    table.append(res2)
    table.append(["", "", "", "", "", ""])
print(tabulate(table))
print(tabulate(table, tablefmt="latex"))


table = []
for i in range(len(countOfData)):
    strS = "pois n = " + str(countOfData[i])
    table.append([strS, "", "", "", "", ""])
    table.append(["", "x", "med x", "zR", "zQ", "zTr"])
    res1 = ["E(z)"]
    res2 = ["D(z)"]
    for j in range(countOfValues):
        res = Disp(Pvalues[i * countOfValues + j])
        res1.append(res[0])
        res2.append(res[1])
    table.append(res1)
    table.append(res2)
    table.append(["", "", "", "", "", ""])
print(tabulate(table))
print(tabulate(table, tablefmt="latex"))


table = []
for i in range(len(countOfData)):
    strS = "unif n = " + str(countOfData[i])
    table.append([strS, "", "", "", "", ""])
    table.append(["", "x", "med x", "zR", "zQ", "zTr"])
    res1 = ["E(z)"]
    res2 = ["D(z)"]
    for j in range(countOfValues):
        res = Disp(Uvalues[i * countOfValues + j])
        res1.append(res[0])
        res2.append(res[1])
    table.append(res1)
    table.append(res2)
    table.append(["", "", "", "", "", ""])
print(tabulate(table))
print(tabulate(table, tablefmt="latex"))