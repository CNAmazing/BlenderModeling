import bpy

# obj = bpy.context.scene.objects.get("Cube")  
# bpy.context.view_layer.objects.active = obj
# bpy.ops.object.mode_set(mode='EDIT')
# # 打开 UV 编辑器并展开 UV
# # 选择所有面
# # bpy.ops.mesh.select_all(action='SELECT')
# uv_layer = obj.data.uv_layers.new(name="Custom_UV_Layer")
# bpy.ops.uv.lightmap_pack( 
#     PREF_CONTEXT='ALL_FACES',
# )  

# obj.data.uv_layers.active = uv_layer
# bpy.ops.object.mode_set(mode='OBJECT')
# uv_layer = obj.data.uv_layers.active.data
# print(f"uv_layer:{uv_layer}")  

# face_index = 0  # 替换为你要修改的面的索引
# face = obj.data.polygons[face_index]
# # 获取该面的顶点索引
# vertex_indices = face.vertices
# # 修改每个顶点的UV坐标
# for i, vertex_index in enumerate(vertex_indices):
#     # 获取当前顶点对应的UV坐标
#     uv = uv_layer[face.loop_start + i].uv
#     print(f"face.loop_start:{face.loop_start}")
#     print(f"face.loop_start:{face.loop_total}")
#     # 修改UV坐标，例如，将UV坐标的X轴值增加0.1，Y轴值增加0.1
#     uv.x += 0.1
#     uv.y += 0.1

# # 返回对象模式
# bpy.ops.object.mode_set(mode='OBJECT')
import sys

# 获取 Blender 自带的 Python 解释器路径
blender_python_path = getattr(bpy.app, "binary_path_python", None)
if blender_python_path:
    print("Blender's Python Path:", blender_python_path)
else:
    print("Blender's Python Path not found.")

# 获取当前 Python 解释器路径
current_python_path = sys.executable
print("Current Python Path:", current_python_path)

# 获取 Blender 可执行文件路径
blender_path = bpy.app.binary_path
print("Blender Executable Path:", blender_path)
import matplotlib.pyplot as plt
import numpy as np

# 创建一个图形和子图
fig, ax = plt.subplots()

# 定义圆的参数
radius = 1  # 半径
center = (0, 0)  # 圆心坐标

# 生成圆的点
theta = np.linspace(0, 2 * np.pi, 100)  # 角度从 0 到 2π
x = center[0] + radius * np.cos(theta)  # x 坐标
y = center[1] + radius * np.sin(theta)  # y 坐标

# 绘制圆形
ax.plot(x, y, label="Circle")

# 设置图形属性
ax.set_aspect('equal')  # 设置 x 和 y 轴比例相同
ax.grid(True)  # 显示网格
ax.axhline(0, color='black', linewidth=0.5)  # 绘制 x 轴
ax.axvline(0, color='black', linewidth=0.5)  # 绘制 y 轴
ax.legend()  # 显示图例

# 显示图形
plt.savefig("output.png")