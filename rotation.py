import numpy as np
# from scipy.spatial.transform import Rotation as R

def solve_transformation(a_points, b_points):
    """
    求解从 a 坐标系到 b 坐标系的旋转矩阵和平移向量
    :param a_points: a 坐标系下的 3 个点，形状为 (3, 3)
    :param b_points: b 坐标系下的 3 个点，形状为 (3, 3)
    :return: 旋转矩阵 R 和平移向量 T
    """
    # 计算 a_points 和 b_points 的质心（centroid）
    a_centroid = np.mean(a_points, axis=0)
    b_centroid = np.mean(b_points, axis=0)
    
    # 将点移动到质心
    a_points_centered = a_points - a_centroid
    b_points_centered = b_points - b_centroid
    
    # 计算旋转矩阵 R：使用 SVD（奇异值分解）
    H = np.dot(a_points_centered.T, b_points_centered)  # 计算协方差矩阵
    U, _, Vt = np.linalg.svd(H)  # 对协方差矩阵进行奇异值分解
    
    # 旋转矩阵 R = Vt.T * U.T
    R_matrix = np.dot(Vt.T, U.T)
    
    # 确保旋转矩阵是右手坐标系
    if np.linalg.det(R_matrix) < 0:
        Vt[2, :] *= -1
        R_matrix = np.dot(Vt.T, U.T)
    
    # 计算平移向量 T
    T_vector = b_centroid - np.dot(a_centroid, R_matrix)
    
    return R_matrix, T_vector
def convert_point_to_b(P_a, R_matrix, T_vector):
    """
    将 a 坐标系下的点 P_a 转换到 b 坐标系下
    :param P_a: a 坐标系下的点 (3,)
    :param R_matrix: 旋转矩阵 R (3, 3)
    :param T_vector: 平移向量 T (3,)
    :return: b 坐标系下的点 (3,)
    """
    P_b = np.dot(R_matrix, P_a) + T_vector
    return P_b
# 示例：3个已知点
a_points = np.array([[1, 0, 1], [1, 1, 1], [1, 0, 0]])  # a 坐标系下的点
b_points = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])  # b 坐标系下的点

# 调用函数求解
R_matrix, T_vector = solve_transformation(a_points, b_points)

print("旋转矩阵 R:")
print(R_matrix)

print("\n平移向量 T:")
print(T_vector)
P_a = np.array([1, 1, 0])
# 转换到 b 坐标系
P_b = convert_point_to_b(P_a, R_matrix, T_vector)

print("a 坐标系下的点 P_a:", P_a)
print("b 坐标系下的点 P_b:", P_b)