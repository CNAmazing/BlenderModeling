import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 定义矩形数据（左上角坐标 x, y，宽度 width，高度 height）
rectangles = [
    (1, 1, 1, 3),  
    (3, 1, 1, 3),  
    (5, 1, 1, 3),  
    (7, 1, 1, 3),  
    (1, 5, 1, 3),  
    (3, 5, 1, 3),  
    (5, 5, 1, 3),  
    (7, 5, 1, 3), 
    (4, 9, 3, 1), 
]

# 创建绘图
fig, ax = plt.subplots()

# 设置坐标轴范围
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# 反转 y 轴，使 (0,0) 在左下角，符合常见坐标系习惯
ax.invert_yaxis()

# 绘制矩形
for (x, y, width, height) in rectangles:
    # matplotlib 的 Rectangle 是以左下角为基准，所以需要调整 y 坐标
    rect = patches.Rectangle((x, y ), width, height, linewidth=2, edgecolor='blue', facecolor='cyan', alpha=0.5)
    ax.add_patch(rect)

# 添加网格和标签
ax.grid(True)
ax.set_xlabel('X 轴')
ax.set_ylabel('Y 轴')
ax.set_title('10x10 平面上的矩形绘制')

# 显示图形
plt.show()
