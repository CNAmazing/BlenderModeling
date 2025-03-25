import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from skopt import gp_minimize
from skopt.space import Categorical
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(42)
data, _ = make_blobs(n_samples=100, centers=4, cluster_std=1.0, random_state=42)

# 使用 K-Means 进行聚类
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(data)

# 可视化聚类结果
plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k')
plt.title("Initial Clustering")
plt.show()

def objective(group_assignment, clusters, data):
    """
    目标函数：最大化组内相似度，最小化组间差异。
    group_assignment: 每个聚类分配到哪个组（例如 [0, 1, 0, 1] 表示 4 个聚类分配到 2 个组）。
    clusters: 聚类结果（每个样本的聚类标签）。
    data: 原始数据（用于计算相似度）。
    """
    n_clusters = len(np.unique(clusters))
    n_groups = len(np.unique(group_assignment))
    
    # 将样本分配到组
    group_labels = np.array([group_assignment[cluster] for cluster in clusters])
    
    # 计算组内相似度
    intra_group_similarity = 0
    for group in range(n_groups):
        group_indices = np.where(group_labels == group)[0]
        if len(group_indices) > 1:
            group_data = data[group_indices]
            distances = pairwise_distances(group_data)
            intra_group_similarity += -np.mean(distances)  # 负号表示最大化相似度
    
    # 计算组间差异
    inter_group_difference = 0
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            group_i_indices = np.where(group_labels == i)[0]
            group_j_indices = np.where(group_labels == j)[0]
            if len(group_i_indices) > 0 and len(group_j_indices) > 0:
                group_i_data = data[group_i_indices]
                group_j_data = data[group_j_indices]
                distances = pairwise_distances(group_i_data, group_j_data)
                inter_group_difference += np.mean(distances)  # 最大化组间差异
    
    # 综合目标函数
    return -(intra_group_similarity + inter_group_difference)  # 负号表示最小化
n_clusters = 4  # 聚类数量
k = 2  # 分组数量

# 定义搜索空间：每个聚类分配到哪个组
space = [Categorical([0, 1], name=f'group_{i}') for i in range(n_clusters)]
# 定义目标函数（固定 clusters 和 data）
def wrapped_objective(group_assignment):
    return objective(group_assignment, clusters, data)

# 执行贝叶斯优化
result = gp_minimize(
    wrapped_objective,
    space,
    n_calls=20,  # 目标函数调用次数
    random_state=42
)

# 输出结果
print("最优分组方案:", result.x)
print("最优目标函数值:", result.fun)

# 将最优分组方案应用于数据
best_group_assignment = result.x
group_labels = np.array([best_group_assignment[cluster] for cluster in clusters])

# 可视化分组结果
plt.scatter(data[:, 0], data[:, 1], c=group_labels, cmap='viridis', marker='o', edgecolor='k')
plt.title("Optimal Grouping")
plt.show()