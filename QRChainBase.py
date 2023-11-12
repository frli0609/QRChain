import numpy as np
import scipy
import sympy
import matplotlib.pyplot as plt
import pandas as pd

M = 8
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
        if 3 * i <= j < 3 * i + 3:
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
        if i % 3 == 0 and i <= j < (i + 3):
            R_r_great[i][j] = R_r[0][j % 3]
            # j += 1
        elif i % 3 == 1 and (i - 1) <= j < (i + 2):
            R_r_great[i][j] = R_r[1][j % 3]
            # j += 1
        elif i % 3 == 2 and (i - 2) <= j < (i + 1):
            R_r_great[i][j] = R_r[2][j % 3]
            # j += 1
        else:
            R_r_great[i][j] = 0
            # j += 1
    # i += 1
# 计算过程参数S_3^k
S_3_1 = -np.dot(N_prime, R_r_great)

# 雅可比矩阵求解
J = np.zeros((6, 6), dtype=np.float64)
for i in range(6):
    J[i][:] = np.hstack((-normal1_T[i], np.cross(normal1_T[i].T, loc1_T[i].T)))
print(J)
# 求解系数参数S_5^K，用雅可比矩阵的广义逆矩阵，当零件的位姿完全确定时，J是非异奇的，可以求逆
S_5_1 = np.linalg.pinv(J)
print(S_5_1)

# 定义S_6^k计算过程参数和计算系数矩阵S_6^k
H0_f3_f1 = np.array([
    [-1, 0, 0, 0],
    [0, 1, 0, -12.5],
    [0, 0, -1, -120],
    [0, 0, 0, 1]
])

H0_f3_R = H0_f3_f1

# 提取旋转矩阵和平移向量
R_f3_R = H0_f3_R[0:3, 0:3]
t_f3_R = H0_f3_R[0:3, 3]

# print(R_f3_to_R)
# print(t_f3_to_R)

# 定义旋转矩阵的转置
R_f3_R_T = R_f3_R.T

# 定义平移向量的叉积矩阵
t_hat_f3_R = np.array([
    [0, t_f3_R[2], -t_f3_R[1]],
    [-t_f3_R[2], 0, t_f3_R[0]],
    [t_f3_R[1], -t_f3_R[0], 0]
])

# 计算 S_6^1 矩阵
S_6_1_upper = np.dot(-R_f3_R_T, t_hat_f3_R)
S_6_1 = np.block([
    [-R_f3_R_T, S_6_1_upper],
    [np.zeros((3, 3)), -R_f3_R_T]
])
print(S_6_1)

# 构建S_8^k矩阵
# 创建全零矩阵
S_8_1 = np.zeros((48, 6))

# 在12-18行上设置单位矩阵
S_8_1[12:18, :] = np.eye(6)
# print(S_8_1)

# 创建列向量 U_f^1
U_f_1 = np.array(
    [[0.1], [0], [-0.1], [0], [0], [0.05], [0], [-0.2], [0], [0], [0], [-0.1], [0.05], [0], [0.1], [0.1], [0], [0]])

# 创建列向量 U_m^1 或 x_j
x_j = np.array([[0], [0], [0.0150], [0], [0], [0]])
U_m_1 = x_j

# 执行计算 x_f3'
x_f3_prime = S_6_1.dot(S_5_1).dot(S_3_1).dot(U_f_1) + U_m_1
# print(x_f3_prime)

# 计算第一项 S_8^1 * S_6^1 * S_5^1 * S_3^1 * U_f^1
first_term = S_8_1.dot(S_6_1).dot(S_5_1).dot(S_3_1).dot(U_f_1)

# 计算第二项 S_8^1 * U_m^1
second_term = S_8_1.dot(U_m_1)

# 合并两项得到 X_1
X_1 = first_term + second_term
# print(X_1)

# =================================================工序2======================================================
# 各个特征相对于RCS的齐次变换矩阵
H0_f_R = []
H0_f1_R = np.array([
    [-1, 0, 0, 0],
    [0, 1, 0, 12.5],
    [0, 0, -1, -120],
    [0, 0, 0, 1]
])

H0_f2_R = np.array([
    [-1, 0, 0, 0],
    [0, 1, 0, 12.5],
    [0, 0, -1, -110],
    [0, 0, 0, 1]
])

H0_f3_R = np.zeros((4,4), dtype=np.float64)

H0_f4_R = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 125],
    [0, 0, 1, -20],
    [0, 0, 0, 1]
])

H0_f5_R = np.array([
    [0.9701, 0, -0.2425, 0],
    [-0.2425, 0, -0.9701, -112.5],
    [0, 1, 0, -60],
    [0, 0, 0, 1]
])

H0_f6_R = np.array([
    [0.3162, 0, -0.9487, -80],
    [0, 1, 0, 0],
    [0.9487, 0, 0.3162, -60],
    [0, 0, 0, 1]
])

H0_f7_R = np.array([
    [0.9701, 0, -0.2425, 0],
    [0.2425, 0, 0.9701, 137.5],
    [0, -1, 0, -60],
    [0, 0, 0, 1]
])

H0_f8_R = np.array([
    [0.3162, 0, 0.9487, 80],
    [0, 1, 0, 0],
    [-0.9487, 0, 0.3162, -60],
    [0, 0, 0, 1]
])

H0_f_R.append(H0_f1_R)
H0_f_R.append(H0_f2_R)
H0_f_R.append(H0_f3_R)
H0_f_R.append(H0_f4_R)
H0_f_R.append(H0_f5_R)
H0_f_R.append(H0_f6_R)
H0_f_R.append(H0_f7_R)
H0_f_R.append(H0_f8_R)

# 定义从状态转移矩阵H21到误差变换矩阵Q21之间的函数
def compute_Q_from_H(H):
    """
    Compute the Q matrix from a given homogeneous transformation matrix H.
    
    Args:
    H (numpy.ndarray): A 4x4 homogeneous transformation matrix.
    
    Returns:
    numpy.ndarray: The computed Q matrix.
    """
    # 从状态转移方程提取旋转矩阵和平移量
    R = H[:3, :3]
    t = H[:3, 3]

    # 反斜对称阵
    skew_t = np.array([[0, -t[2], t[1]],
                       [t[2], 0, -t[0]],
                       [-t[1], t[0], 0]])

    # Q矩阵计算
    Q_upper = np.hstack((R.T, -R.T @ skew_t))
    Q_lower = np.hstack((np.zeros((3, 3)), R.T))
    Q = np.vstack((Q_upper, Q_lower))

    return Q

#需要知道已经加工完成的特征、上一道工序的基准、这一道工序的基准（这一个工序的RCS与哪个特征重合）、下一道工序及以后才会加工的特征分别是哪些*********************************************
Q = []
for i in range(M):
    if i+1 == 3 or i+1 == 4:     ########这里的3和4以后要用R和F代替（当前基准特征和尚未加工的特征）
        Q.append(np.zeros((6,6), dtype=np.float64))
    else: 
        Q.append(compute_Q_from_H(H0_f_R[i]))
print(Q)


def create_error_transformation_matrix(M, R, Q_matrices):
    """
    Create the error transformation matrix S_1^k.

    Parameters:
    M (int): The total number of features.
    R (int): The center of transformation (1-based index).
    Q_matrices (list of np.array): List of M 6x6 Q matrices.

    Returns:
    np.array: The error transformation matrix S_1^k.
    """
    # Initialize a 6M x 6M zero matrix
    S = np.zeros((6*M, 6*M))

    # Fill the diagonal blocks with 6x6 identity matrices
    for i in range(M):
        S[6*i:6*(i+1), 6*i:6*(i+1)] = np.eye(6)

    # Fill the non-diagonal blocks in the R-th row with Q matrices
    for i in range(M):
        if i != R - 1:
            S[6*i:6*(i+1),6*(R-1):6*R] = -Q_matrices[i]

    return S

# Example usage
R = 3
# Compute the error transformation matrix
S_1_k = create_error_transformation_matrix(M, R, Q )