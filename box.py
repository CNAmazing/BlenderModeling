import numpy as np
import cv2
from heapq import heappush, heappop

class TextureSynthesizer:
    def __init__(self, source_image):
        self.source_image = source_image
        self.height, self.width = source_image.shape[:2]
        self.cuts = self.find_parallel_cuts()

    def find_parallel_cuts(self):
        # 这里我们简化了切割的生成过程
        # 在实际实现中，应该使用动态规划来找到最小成本的切割
        cuts = []
        for x in range(self.width - 1):
            cut = [(x, y) for y in range(self.height)]
            cuts.append(cut)
        return cuts

    def compute_cost(self, cut1, cut2):
        # 计算两个切割之间的颜色差异成本
        cost = 0
        for (x1, y1), (x2, y2) in zip(cut1, cut2):
            cost += np.linalg.norm(self.source_image[y1, x1] - self.source_image[y2, x2])
        return cost

    def synthesize_texture(self, target_width):
        # 构建图并找到最短路径
        graph = self.build_graph()
        start_node = (0, 0)
        end_node = (target_width, 0)
        path = self.dijkstra(graph, start_node, end_node)
        return self.reconstruct_image(path)

    def build_graph(self):
        graph = {}
        for i, cut1 in enumerate(self.cuts):
            for j, cut2 in enumerate(self.cuts):
                if i != j:
                    cost = self.compute_cost(cut1, cut2)
                    graph[(i, 0)] = graph.get((i, 0), []) + [((j, cost), cost)]
        return graph

    def dijkstra(self, graph, start, end):
        queue = []
        heappush(queue, (0, start, []))
        visited = set()
        while queue:
            (cost, node, path) = heappop(queue)
            if node not in visited:
                visited.add(node)
                path = path + [node]
                if node == end:
                    return path
                for (neighbor, edge_cost) in graph.get(node, []):
                    if neighbor not in visited:
                        heappush(queue, (cost + edge_cost, neighbor, path))
        return []

    def reconstruct_image(self, path):
        # 根据路径重建图像
        new_image = np.zeros((self.height, len(path), 3), dtype=np.uint8)
        for i, (cut_index, _) in enumerate(path):
            cut = self.cuts[cut_index]
            for (x, y) in cut:
                new_image[y, i] = self.source_image[y, x]
        return new_image

# 示例使用
source_image = cv2.imread('E:/Desktop/Simulationtest/Simulationtest/background.png')
synthesizer = TextureSynthesizer(source_image)
new_texture = synthesizer.synthesize_texture(target_width=1600)
cv2.imwrite('synthesized_texture.jpg', new_texture)