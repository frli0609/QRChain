import numpy as np
import scipy
import sympy
import matplotlib.pyplot as plt
import pandas as pd

# S_4^k计算需要的矩阵基础算子
sigma_1 = np.array([[1],
                    [0],
                    [0],
                    [0]])
sigma_2 = np.array([[0],
                   [1],
                   [0],
                   [0]])
sigma_3 = np.array([[0],
                    [0],
                    [1],
                    [0]])
sigma_4 = np.array([[0, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0]])
sigma_5 = np.array([[0, 0, 1, 0],
                    [0, 0, 0, 0],
                    [-1, 0, 0, 0],
                    [0, 0, 0, 0]])
sigma_6 = np.array([[0, -1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])

# 声明状态空间参数
X_0 = np.zeros((48, 1))
X_1 = np.zeros((48, 1))

# 工序1
loc1_T = np.array([[50, 0, 0],
                   [-50, -100, 0],
                   [-50, 100, 0],
                   [80, -80, -60],
                   [80, 80, -60],
                   [0, -125, -60]])

normal1_T = np.array([[0, 0, 1],
                      [0, 0, 1],
                      [0, 0, 1],
                      [0.9487, 0, -0.3162],
                      [0.9487, 0, -0.3162],
                      [0.2425, -0.9701, 0]])

error1_T = np.array([[0.1, 0, -0.1],
                     [0, 0, 0.05],
                     [0, -0.2, 0],
                     [0, 0, -0.1],
                     [0.05, 0, 0.1],
                     [0.1, 0, 0]])

# 工序2
loc2_T = np.array([[-50, 0, 0],
                   [50, 100, 0],
                   [50, -100, 0],
                   [-80, 80, -60],
                   [-80, -80, -60],
                   [0, 125, -60]])

normal2_T = np.array([[0, 0, 1],
                      [0, 0, 1],
                      [0, 0, 1],
                      [-0.9487, 0, 0.3162],
                      [-0.9487, 0, 0.3162],
                      [-0.2425, 0.9701, 0]])

error2_T = np.array([[0, 0, 0.15],
                     [0.1, 0, -0.1],
                     [0.1, 0, 0],
                     [0, -0.3, 0.1],
                     [0, -0.3, 0],
                     [0, 0, 0.05]])

# 工序3
loc3_T = np.array([[50, 0, 0],
                   [-50, -100, 0],
                   [-50, 100, 0],
                   [80, -80, -50],
                   [80, 80, -50],
                   [0, -125, -50]])

normal3_T = np.array([[0, 0, 1],
                      [0, 0, 1],
                      [0, 0, 1],
                      [0.9487, 0, -0.3162],
                      [0.9487, 0, -0.3162],
                      [0.2425, -0.9701, 0]])

error3_T = np.array([[0.2, 0, 0.1],
                     [0, 0, -0.05],
                     [0, -0.2, 0],
                     [0, 0, -0.1],
                     [0, 0, 0.1],
                     [0, 0, 0.1]])

# 计算N'矩阵，等于n_i\prime的转置组合成的对角矩阵
N_prime = np.zeros((6, 18), dtype=np.float64)
for i in range(6):
    for j in range(18):
        if 3*i <= j < 3*i+3:
          N_prime[i][j] = normal1_T[i][j % 3]
        else:
          N_prime[i][j] = 0
# print(N_prime)

# 计算R_R矩阵
R_r = np.identity(3)
# print(R_r)

# 生成R_R长对角矩阵
R_r_great = np.zeros((18, 18), dtype=np.float64)
for i in range(18):
    for j in range(18):
        if i % 3 == 0 and i <= j < (i+3):
            R_r_great[i][j] = R_r[0][j % 3]
            # j += 1
        elif i % 3 == 1 and (i-1) <= j < (i+2):
            R_r_great[i][j] = R_r[1][j % 3]
            # j += 1
        elif i % 3 == 2 and (i-2) <= j < (i+1):
            R_r_great[i][j] = R_r[2][j % 3]
            # j += 1
        else:
            R_r_great[i][j] = 0
            # j += 1
    # i += 1
### 计算过程参数S_3^k
S_3_1 = -np.dot(N_prime,R_r_great)

