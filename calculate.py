import bpy
import numpy as np
def plane_equation(normal_vector, point):
    """
    计算三维平面方程 Ax + By + Cz + D = 0
    :param normal_vector: (A, B, C) - 平面的法向量
    :param point: (x0, y0, z0) - 平面上的一点
    :return: 平面方程系数 (A, B, C, D)
    """
    A, B, C = normal_vector
    x0, y0, z0 = point
    
    # 计算 D
    D = -(A * x0 + B * y0 + C * z0)
    
    return A, B, C, D
def is_point_on_plane(point, plane_coeffs):
    """
    判断一个点是否在平面上
    :param point: (x, y, z) - 需要判断的点
    :param plane_coeffs: (A, B, C, D) - 平面方程系数
    :return: True (在平面上) 或 False (不在平面上)
    """
    x, y, z = point
    A, B, C, D = plane_coeffs
    
    # 判断 Ax + By + Cz + D 是否等于 0
    return A * x + B * y + C * z + D == 0

def are_vectors_orthogonal(vec1, vec2):
    """
    判断两个向量是否正交
    :param vec1: 第一个向量 (x1, y1, z1)
    :param vec2: 第二个向量 (x2, y2, z2)
    :return: 如果两个向量正交，返回 True，否则返回 False
    """
    # 计算两个向量的点积
    dot_product = np.dot(vec1, vec2)

    # 如果点积为 0，说明两个向量正交
    return dot_product == 0

def create_material(color_dict):
    """
    创建一个新材质并将其添加到物体
    :param obj: 物体
    :param material_name: 材质名称
    :param color: 颜色元组 (R, G, B)
    """
    # 创建新材质
    for key,value in color_dict.items():
        material_name=key
        R,G,B=value
        R = R / 255
        G = G / 255
        B = B / 255
        material = bpy.data.materials.new(name=material_name)
        material.diffuse_color = (R, G, B, 1)
def add_material_to_object(obj,color_dict):       
    for key,value in color_dict.items():
        material_name=key
        material = bpy.data.materials.get(material_name)
        if material.name not in obj.data.materials:
            obj.data.materials.append(material)
def add_material_by_faceIdx(obj,faceIdx,material_name):
    """
    为物体的指定面添加材质
    :param object: 物体
    :param faceIdx: 面的索引
    :param material_name: 材质名称
    """
    # 确保物体处于编辑模式
    bpy.ops.object.mode_set(mode='EDIT')
    # 获取网格数据
    mesh = obj.data
    # 切换到对象模式以访问面片数据
    bpy.ops.object.mode_set(mode='OBJECT')
    # 获取材质索引
    material_index = obj.data.materials.find(material_name)
    # 获取指定面片（例如索引为0的面片）
    mesh.polygons[faceIdx].material_index = material_index
    # 切换回物体模式
    bpy.ops.object.mode_set(mode='OBJECT')
    
point=np.array([5,5,5])
normal=np.array([1,0,0])
plane_coeffs = plane_equation(normal, point)
print(plane_coeffs)
plane_face=[]
side_face=[]
obj = bpy.context.scene.objects.get("Cube")
if obj.type == 'MESH':
    mesh = obj.data
    # 获取物体的变换矩阵
    for poly in obj.data.polygons:
        # 获取面片索引
        print(f"面片索引: {poly.index}")
        # 获取面片的顶点索引
        vertices = [obj.data.vertices[i] for i in poly.vertices]
        # 将局部空间坐标转换为世界空间坐标
        world_vertices = [obj.matrix_world @ v.co for v in vertices]
        print(f"顶点坐标: {world_vertices}")
        # 判断面片是否在平面上
        for i,point in enumerate(world_vertices):
            print(f"point:{point}")                
            if not is_point_on_plane(point, plane_coeffs):
                break
            if i == len(point) - 1:
                plane_face.append(poly.index)
        #判断面片的法向量是否与平面法向量正交 且顶点在平面上
        for i, point in enumerate(world_vertices):
            if is_point_on_plane(point, plane_coeffs) and are_vectors_orthogonal(poly.normal, normal):
                side_face.append(poly.index)
                break
else:
    print("当前对象不是网格类型")

color_dict={'facade':(0,255,0),
             'windows':(255,0,0),
             'glass':(0,0,255),
             'door':(247,200,60),
             'balcony':(32,140,62),}

create_material(color_dict)
add_material_to_object(obj,color_dict)

# 获取指定面片（例如索引为0的面片）
for face_index in plane_face:
    add_material_by_faceIdx(obj,face_index,"facade")
for side_face_index in side_face:
    add_material_by_faceIdx(obj,side_face_index,"windows")
    
# 切换回物体模式
# bpy.ops.object.mode_set(mode='OBJECT')
print(f"side_face:{side_face}")

