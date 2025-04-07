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
# import sys

# # 获取 Blender 自带的 Python 解释器路径
# blender_python_path = getattr(bpy.app, "binary_path_python", None)
# if blender_python_path:
#     print("Blender's Python Path:", blender_python_path)
# else:
#     print("Blender's Python Path not found.")

# # 获取当前 Python 解释器路径
# current_python_path = sys.executable
# print("Current Python Path:", current_python_path)

# # 获取 Blender 可执行文件路径
# blender_path = bpy.app.binary_path
# print("Blender Executable Path:", blender_path)
# import matplotlib.pyplot as plt
# import numpy as np

# # 创建一个图形和子图
# fig, ax = plt.subplots()

# # 定义圆的参数
# radius = 1  # 半径
# center = (0, 0)  # 圆心坐标

# # 生成圆的点
# theta = np.linspace(0, 2 * np.pi, 100)  # 角度从 0 到 2π
# x = center[0] + radius * np.cos(theta)  # x 坐标
# y = center[1] + radius * np.sin(theta)  # y 坐标

# # 绘制圆形
# ax.plot(x, y, label="Circle")

# # 设置图形属性
# ax.set_aspect('equal')  # 设置 x 和 y 轴比例相同
# ax.grid(True)  # 显示网格
# ax.axhline(0, color='black', linewidth=0.5)  # 绘制 x 轴
# ax.axvline(0, color='black', linewidth=0.5)  # 绘制 y 轴
# ax.legend()  # 显示图例

# # 显示图形
# plt.savefig("output.png")


# import os
# import sys
# print("当前工作目录:", cwd)
# # script_path = os.path.abspath(__file__)
# cwd = os.getcwd()
# sys.path.append(cwd)

# print("当前Python模块搜索路径 (sys.path):")
# for path in sys.path:
#     print(path)

import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
含高斯模糊的仿射变换
"""
# def areaA_to_areaB_AffineTransform_By_points(pts_a, pts_b, a, b):
#     M = cv2.getAffineTransform(pts_a, pts_b)
#     h_b, w_b = b.shape[:2]

#     # 1. 将输入图像转换为 float32 类型，避免类型不匹配
#     a_float = a.astype(np.float32)
#     b_float = b.astype(np.float32)

#     # 2. 使用更高质量的插值方法（如 INTER_CUBIC）
#     transformed_triangle = cv2.warpAffine(a_float, M, (w_b, h_b))

#     # 3. 创建浮点型掩码（0.0~1.0），并做高斯模糊使边缘平滑
#     mask = np.zeros((h_b, w_b), dtype=np.float32)
#     cv2.fillConvexPoly(mask, np.int32(pts_b), 1.0)
#     mask = cv2.GaussianBlur(mask, (3, 3), 0)  # 边缘模糊，使过渡更自然
#     mask = np.clip(mask, 0, 1)  # 确保值在 [0, 1] 范围内
#     mask_3ch = cv2.merge([mask, mask, mask])  # 扩展为 3 通道

#     # 4. 使用乘法混合（显式指定 dtype=np.float32）
#     transformed_part = cv2.multiply(transformed_triangle, mask_3ch, dtype=cv2.CV_32F)
#     background_part = cv2.multiply(b_float, 1.0 - mask_3ch, dtype=cv2.CV_32F)
#     blended = cv2.add(transformed_part, background_part)

#     # 5. 转换回 uint8 类型（如果原图是 8 位）
#     blended = np.clip(blended, 0, 255).astype(np.uint8)

#     return blended
"""
无高斯模糊的仿射变换
"""
def areaA_to_areaB_AffineTransform_By_points(pts_a, pts_b, a, b):
    M = cv2.getAffineTransform(pts_a, pts_b)
    h_b, w_b = b.shape[:2]

    # 1. 将输入图像转换为 float32 类型，避免类型不匹配
    a_float = a.astype(np.float32)
    b_float = b.astype(np.float32)

    transformed_triangle = cv2.warpAffine(a_float, M, (w_b, h_b))

    # 3. 创建浮点型掩码（0.0~1.0），不进行高斯模糊
    mask = np.zeros((h_b, w_b), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(pts_b), 1.0)
    mask = np.clip(mask, 0, 1)  # 确保值在 [0, 1] 范围内
    mask_3ch = cv2.merge([mask, mask, mask])  # 扩展为 3 通道

    # 4. 使用乘法混合（显式指定 dtype=np.float32）
    transformed_part = cv2.multiply(transformed_triangle, mask_3ch, dtype=cv2.CV_32F)
    background_part = cv2.multiply(b_float, 1.0 - mask_3ch, dtype=cv2.CV_32F)
    blended = cv2.add(transformed_part, background_part)

    # 5. 转换回 uint8 类型（如果原图是 8 位）
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended

imageA=cv2.imread(r'E:\WorkSpace\BlenderModeling\output\00472\images\poly0.jpg')
imageA=cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)  # 将图像 A 转换为 RGB 格式
imageB=np.zeros((imageA.shape[0],imageA.shape[1],3),dtype=np.uint8)  # 创建一个与图像 A 相同大小的空白图像 B
pts_a = np.array([[0, 0], [300, 0], [0, 300]], dtype=np.float32)  # 图像 a 中的三角形顶点
pts_b = np.array([[0, 0], [300, 0], [0, 300]], dtype=np.float32)  # 图像 b 中的三角形顶点

result=areaA_to_areaB_AffineTransform_By_points(pts_a,pts_b,imageA,imageB)
pts_a = np.array([ [300, 0], [0, 300],[300, 300]], dtype=np.float32)  # 图像 a 中的三角形顶点
pts_b = np.array([[300, 0], [0, 300],[300, 300]], dtype=np.float32)  # 图像 b 中的三角形顶点
result=areaA_to_areaB_AffineTransform_By_points(pts_a,pts_b,imageA,result)
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(imageA)
plt.title("Image A")
plt.subplot(1, 3, 2)
plt.imshow(imageB)
plt.title("Image B")
plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title("Image B")
plt.show()