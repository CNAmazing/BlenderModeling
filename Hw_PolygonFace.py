import bpy
import random
import math
import cv2
import numpy as np  
import json

from BlenderScripts import *

try:
    from Hw_Tools import *
except ImportError:
    pass
def areaA_to_areaB_AffineTransform_By_points(pts_a,pts_b,a,b):
    M = cv2.getAffineTransform(pts_a, pts_b)

    # 获取图像 b 的尺寸
    h_b, w_b = b.shape[:2]

    # 应用仿射变换，将图像 a 中的三角形区域变换到图像 b 的三角形区域
    transformed_triangle = cv2.warpAffine(a, M, (w_b, h_b))

    # 创建一个掩码，用于提取变换后的三角形区域
    mask = np.zeros((h_b, w_b), dtype=np.uint8)  # 单通道掩码
    cv2.fillConvexPoly(mask, np.int32(pts_b), 255)  # 在图像 b 的三角形区域创建掩码

    # 将掩码扩展为 3 通道，以便与 transformed_triangle 进行按位与操作
    mask_3ch = cv2.merge([mask, mask, mask])

    # 将变换后的三角形区域融合到图像 b 中
    result = cv2.bitwise_and(transformed_triangle, mask_3ch)  # 提取变换后的三角形区域
    b = cv2.add(b, result)  # 将结果添加到图像 b 中
    return b
# 获取纹理图像的尺寸
def save_texture_for_faces(mesh, face_indices, texture_image, output_path):
    width, height = texture_image.size
    texture_pixels = np.array(texture_image.pixels).reshape((height, width, 4))
    # print("texture_pixels:",texture_pixels)
    # 初始化合并图像的尺寸和背景
    u_min_all = 1.0
    u_max_all = 0.0
    v_min_all = 1.0
    v_max_all = 0.0
    
    # 遍历所有面片，计算 UV 坐标范围
    uv_layer = mesh.uv_layers.active.data
    for face_index in face_indices:
        face = mesh.polygons[face_index]
        uvs = [uv_layer[loop_index].uv for loop_index in range(face.loop_start, face.loop_start + face.loop_total)]
        u_min = min(uv[0] for uv in uvs)
        u_max = max(uv[0] for uv in uvs)
        v_min = min(uv[1] for uv in uvs)
        v_max = max(uv[1] for uv in uvs)
        

        tiny_texture_pixels = texture_pixels[int(v_min * height):int(v_max * height), int(u_min * width):int(u_max * width)]

        

        u_min_all = max(min(u_min_all, u_min),0)
        u_max_all = min(max(u_max_all, u_max),1)
        v_min_all = max(min(v_min_all, v_min),0)
        v_max_all = min(max(v_max_all, v_max),1)
        
    # 计算合并图像的尺寸
    merge_width = int(u_max_all * width) - int(u_min_all * width)
    merge_height = int(v_max_all * height) - int(v_min_all * height)
    
    merge_image = np.zeros((merge_height, merge_width, 4), dtype=np.float32)
    texture_pixels_crop=texture_pixels[int(v_min_all * height):int(v_max_all* height),int(u_min_all * width):int(u_max_all * width)]
    texture_pixels_crop=texture_pixels_crop[::-1]
    mask=np.zeros((merge_height, merge_width), dtype=np.uint8)
    # 遍历所有面片，提取并合并纹理区域
    for face_index in face_indices:
        face = mesh.polygons[face_index]
        uvs = [np.array(uv_layer[loop_index].uv) for loop_index in range(face.loop_start, face.loop_start + face.loop_total)]
        pixel_points = []
        for u,v in uvs:
            x = int(u * width)
            y = int((1 - v) * height)  # Blender 的 V 坐标是反向的
            pixel_points.append([x, y])
        cv2.fillPoly(mask, [np.array(pixel_points)], 1)
        
    mask.astype(dtype=np.bool)
    # merge_image[mask] = texture_pixels_crop[mask]
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                merge_image[i][j]=texture_pixels_crop[i][j]
            else:
                merge_image[i][j]=[0,0,0,1]
    # merge_image[mask] = texture_pixels_crop
    # 将图像从 [0, 1] 范围转换为 [0, 255] 范围
    merge_image_uint8 = (merge_image * 255).astype(np.uint8)
    if merge_image_uint8.size!=0:
        cv2.imwrite(output_path, cv2.cvtColor(merge_image_uint8, cv2.COLOR_RGBA2BGRA))
def get_texture_image_by_object(object: bpy.types.Object):
    material_slots = object.material_slots
    # 遍历材质槽
    for slot in material_slots:
        material = slot.material
        if material:
            # 获取材质节点树
            nodes = material.node_tree.nodes
            # 查找Principled BSDF节点
            for node in nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    # 获取基础颜色输入
                    base_color_input = node.inputs['Base Color']

                    # 如果基础颜色输入连接到纹理节点
                    if base_color_input.is_linked:
                        texture_node = base_color_input.links[0].from_node

                        # 如果纹理节点是图像纹理节点
                        if texture_node.type == 'TEX_IMAGE':
                            texture_image = texture_node.image
                            print(f"Material: {material.name}, Texture Image: {texture_image}")
                            return texture_image
                    break
def point_To_pixelUV(point,origin,actual_Width,actual_Height,poly_Width,poly_Height,R):
    x,y,z=point
    x_axis = R[0]
    y_axis = R[1]
    x = np.dot([x,y,z], x_axis)
    y = np.dot([x,y,z], y_axis)
    x_pixel = (x - origin[0]) / actual_Width* poly_Width
    y_pixel = (y - origin[1]) / actual_Height* poly_Height
    return x_pixel,y_pixel
  
def generate_FacadeTexture(mesh,texture_image,poly_Information):

    width, height = texture_image.size
    poly_Width = poly_Information['poly_ImageSize'][0]
    poly_Height = poly_Information['poly_ImageSize'][1]
    actual_Width = poly_Information['poly_ActualSize'][0]
    actual_Height = poly_Information['poly_ActualSize'][1]
    face_idxs = poly_Information['faces_idx']
    origin = poly_Information['origin']
    R=poly_Information['basis']
    texture_pixels = np.array(texture_image.pixels).reshape((height, width, 4))
    texture_pixels = texture_pixels[::-1]
    texture_pixels = (texture_pixels * 255).astype(np.uint8)

    texture_pixels_bgra = cv2.cvtColor(texture_pixels, cv2.COLOR_RGBA2BGRA)
    texture_pixels_bgr = texture_pixels_bgra[:, :, :3]

    uv_layer_name = "UVMap"
    if uv_layer_name in mesh.uv_layers:
        uv_layer = mesh.uv_layers[uv_layer_name]
        mesh.uv_layers.active = uv_layer
    else:
            # 如果不存在，创建新的 UV 层并激活
        uv_layer = mesh.uv_layers.new(name=uv_layer_name)
        mesh.uv_layers.active = uv_layer
    FacadeTexture=np.zeros((poly_Height, poly_Width,3), dtype=np.uint8)
    for face_idx in face_idxs:
        face = mesh.polygons[face_idx]
        vertex_coords = np.array([np.array(mesh.vertices[vertex_idx].co) for vertex_idx in face.vertices])
        uvs=[]
        for loop_index in face.loop_indices:
            uv=uv_layer.data[loop_index].uv 
            u,v=uv[0],uv[1]
            uvs.append([u,v])
        
        # uvs = np.array([np.array(uv_layer.data[loop_index].uv) for loop_index in face.loop_indices])
        poly_points = []
        for point in vertex_coords:
            x_pixel,y_pixel = point_To_pixelUV( point=point, 
                                                actual_Height=actual_Height,
                                                actual_Width=actual_Width,
                                                origin=origin,
                                                poly_Height=poly_Height,
                                                poly_Width=poly_Width,
                                                R=R)
            poly_points.append([x_pixel, y_pixel])
        uv_points = [[uv[0] * width, uv[1] * height] for uv in uvs]
        poly_points = np.float32(poly_points)                                       
        uv_points = np.float32(uv_points)        
        """
        uv_points: 原始纹理坐标点
        poly_points: 多边形坐标点
        """
        FacadeTexture=areaA_to_areaB_AffineTransform_By_points(uv_points, poly_points,  texture_pixels_bgr,FacadeTexture)
    return FacadeTexture    

        # u_min = min(uv[0] for uv in uvs)
        # u_max = max(uv[0] for uv in uvs)
        # v_min = min(uv[1] for uv in uvs)
        # v_max = max(uv[1] for uv in uvs)
        


        # u_min_all = max(min(u_min_all, u_min),0)
        # u_max_all = min(max(u_max_all, u_max),1)
        # v_min_all = max(min(v_min_all, v_min),0)
        # v_max_all = min(max(v_max_all, v_max),1)

# 确保在对象模式下
bpy.ops.object.mode_set(mode='OBJECT')
# 获取名为“x”的物体
obj = bpy.data.objects.get("Cube")
if obj is None:
    raise ValueError("未找到名为 'Cube' 的物体")
# 确保物体是网格类型
if obj.type != 'MESH':
    raise ValueError("物体不是网格类型")
mesh = obj.data
# 确保物体有材质
if not mesh.materials:
    material = bpy.data.materials.new(name="RandomColor")
    mesh.materials.append(material)

# 遍历每个面片
polygons_Parameterization={}
poly_idx=0
total_face_Set=set() 
for face in mesh.polygons:
    
    current_face_Set=set()
    current_edge_Set=set()
    normal=face.normal
    if not face_select_by_normal(normal) or face.index in total_face_Set:
        continue
    current_face_Set.add(face.index)
    total_face_Set.add(face.index)
    for edge in face.edge_keys:
        current_edge_Set.add(edge)
    last_len=len(current_face_Set)
    current_len=0
    """
    通过区域生长算法，找到所有与当前面片法向量相似的面片
    """
    while last_len!=current_len:
        last_len=len(current_face_Set)
        for f in mesh.polygons:
            if f.index in current_face_Set:
                continue
            for edge in f.edge_keys:
                if edge in current_edge_Set and is_normals_similar(f.normal,normal):
                    current_face_Set.add(f.index)
                    total_face_Set.add(f.index)
                    for e in f.edge_keys:
                        current_edge_Set.add(e)
                    
        current_len=len(current_face_Set)
    
    
    y_axis=np.array([0,0,1])
    z_axis=normal
    x_axis=np.cross(y_axis,z_axis) 
      
    x_axis=x_axis/np.linalg.norm(x_axis)
    y_axis=y_axis/np.linalg.norm(y_axis)
    z_axis = z_axis/np.linalg.norm(z_axis)
    x_min=float("inf")
    x_max=float("-inf")
    y_min=float("inf")
    y_max=float("-inf")
    for c_face in current_face_Set:
        polygon = mesh.polygons[c_face]
        # vertices = [mesh.vertices[idx].co for idx in polygon.vertices]
        for vertex_idx in polygon.vertices:
            x,y,z=mesh.vertices[vertex_idx].co
            u = np.dot([x,y,z], x_axis)
            v = np.dot([x,y,z], y_axis)
            x_min = min(x_min, u)
            x_max = max(x_max, u)
            y_min = min(y_min, v)
            y_max = max(y_max, v)
    actual_width = x_max - x_min
    actual_height = y_max - y_min
    
    if actual_width>actual_height:
        poly_Width=640
        poly_Height=int(poly_Width*actual_height/actual_width)
    else:
        poly_Height=640
        poly_Width=int(poly_Height*actual_width/actual_height)
    
    origin=tuple([x_min,y_min])
    polygons_Parameterization[f'poly{str(poly_idx)}']={
        "origin":origin,
        'poly_ActualSize':(actual_width,actual_height),
        'poly_ImageSize':(poly_Width,poly_Height),
        'basis':tuple([list(x_axis),list(y_axis),list(z_axis)]),
        "faces_idx":tuple(current_face_Set),
        "edges_idx":tuple(current_edge_Set)
    }

    poly_idx+=1
    

print("polygons_Parameterization:",polygons_Parameterization)
texture_image = get_texture_image_by_object(obj)
if texture_image:
    for key,value in polygons_Parameterization.items():
        # save_texture_for_faces(mesh, value['faces_idx'], texture_image, f"{key}.png")
        FacadeTexture=generate_FacadeTexture(mesh,texture_image,value)
        cv2.imwrite(f"{key}.jpg", FacadeTexture)
# with open(f"data.json", "w") as json_file:
#         json.dump(polygons_Parameterization, json_file, indent=4)