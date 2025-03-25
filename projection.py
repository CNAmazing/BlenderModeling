import numpy as np

# 给定一组三维点坐标
points = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [2, 3, 1],
    [6, 7, 8],
])

# 给定法向量
normal = np.array([1, 1, 1])  # 法向量示例

# 计算法向量的长度
normal_length = np.linalg.norm(normal)

# 投影到平面上的函数
def project_to_plane(point, normal):
    # 计算点到平面的距离
    d = np.dot(point, normal) / normal_length
    # 投影到平面
    projection = point - d * normal / normal_length**2
    return projection

# 将所有点投影到平面
projected_points = np.array([project_to_plane(p, normal) for p in points])

# 打印投影后的点
print("投影到平面后的点：")
print(projected_points)
