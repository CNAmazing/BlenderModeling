
#获取当前工作目录 并添加至sys.path
import sys
import os
import bpy
import numpy as np
import cv2
import json
from itertools import combinations
cwd = os.getcwd()
sys.path.append(cwd)
from Hw_Tools import *
def line_plane_intersection(P0, v, Q0, n):
    """
    计算直线与平面的交点
    :param P0: 直线上一点 [x, y, z]
    :param v: 直线方向向量 [dx, dy, dz]
    :param Q0: 平面上一点 [x, y, z]
    :param n: 平面法向量 [a, b, c]
    :return: 交点坐标，或返回是否平行/包含
    """
    P0, v, Q0, n = np.array(P0), np.array(v), np.array(Q0), np.array(n)
    
    denominator = np.dot(n, v)
    numerator = np.dot(n, Q0 - P0)
    
    if np.abs(denominator) < 1e-6:  # 直线与平面平行
        if np.abs(numerator) < 1e-6:
            return "直线在平面内（无限多交点）"
        else:
            return "直线与平面平行（无交点）"
    else:
        t = numerator / denominator
        intersection = P0 + t * v
        return intersection
  
def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"目录 '{path}' 已创建。")
    else:
        print(f"目录 '{path}' 已存在。")
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
# 获取纹理图像的尺寸
def project_points_to_plane(points, n, Q):
    """
    将一组点投影到平面
    :param points: 待投影的点集，形状为 (N, 3) 的数组
    :param n: 平面法向量（无需单位向量，函数内会归一化）
    :param Q: 平面中心点，形状 (3,)
    :return: 投影后的点集，形状 (N, 3)
    """
    n_normalized = n / np.linalg.norm(n)  # 归一化法向量
    Q = np.array(Q)
    points = np.array(points)
    
    # 计算投影
    vectors = points - Q  # 点相对于平面中心的向量
    distances = np.dot(vectors, n_normalized)  # 点乘得到有符号距离
    projected_points = points - distances[:, np.newaxis] * n_normalized
    
    return projected_points
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
        print("vertex_coords:",vertex_coords)
        # vertex_coords_projected = project_points_to_plane(vertex_coords, R[2], center)
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

        poly_points=[[[p_uv[0], poly_Height - p_uv[1]] for p_uv in poly_points]]
        uv_points=[[[p_uv[0], height - p_uv[1]] for p_uv in uv_points]]
        print("poly_points:",poly_points)
        print("uv_points:",uv_points)
        poly_points = np.float32(poly_points)                                       
        uv_points = np.float32(uv_points)  

        """
        uv_points: 原始纹理坐标点
        poly_points: 多边形坐标点
        """
        FacadeTexture=areaA_to_areaB_AffineTransform_By_points(uv_points, poly_points,  texture_pixels_bgr,FacadeTexture)
    return FacadeTexture 
def main():
    output_path = "output"
    obj_name = "00472"
    # 确保在对象模式下
    bpy.ops.object.mode_set(mode='OBJECT')
    # 获取名为“x”的物体
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        raise ValueError(f"未找到名为{obj_name} 的物体")
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
        current_vertex_Set=set()
        normal=face.normal
        center=None
        if not face_select_by_normal(normal) or face.index in total_face_Set:
            continue
        current_face_Set.add(face.index)
        total_face_Set.add(face.index)
        for edge in face.edge_keys:
            current_edge_Set.add(edge)
            current_vertex_Set.add(edge[0])
            current_vertex_Set.add(edge[1])
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
                            current_vertex_Set.add(e[0])
                            current_vertex_Set.add(e[1])
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
   
        
        poly_vertices = np.array([np.array(mesh.vertices[v_idx].co) for v_idx in current_vertex_Set]) 
        print("vertices:",poly_vertices)
        u=np.dot(poly_vertices, x_axis)
        v=np.dot(poly_vertices, y_axis)
        x_min=min(x_min, np.min(u))
        x_max=max(x_max, np.max(u))
        y_min=min(y_min, np.min(v))
        y_max=max(y_max, np.max(v))

        x_min_idx = np.argmin(u)
        x_max_idx = np.argmax(u)
        y_min_idx = np.argmin(v)
        y_max_idx = np.argmax(v)
        x_min_point = poly_vertices[x_min_idx]
        x_max_point = poly_vertices[x_max_idx]
        y_min_point = poly_vertices[y_min_idx]
        y_max_point = poly_vertices[y_max_idx]
        r1=line_plane_intersection(P0=y_min_point,
                                    v=x_axis,
                                    Q0=x_min_point,
                                    n=x_axis)
        r2=line_plane_intersection(P0=y_max_point,
                                    v=x_axis,
                                    Q0=x_max_point,
                                    n=x_axis)
        r3=line_plane_intersection(P0=x_min_point,
                                    v=y_axis,
                                    Q0=y_min_point,
                                    n=y_axis)
        r4=line_plane_intersection(P0=x_max_point,
                                    v=y_axis,
                                    Q0=y_max_point,
                                    n=y_axis)           
        center=(r1+r2+r3+r4)/4
    
        actual_width = x_max - x_min
        actual_height = y_max - y_min
        
        if actual_width>actual_height:
            poly_Width=640
            poly_Height=int(poly_Width*actual_height/actual_width)
        else:
            poly_Height=640
            poly_Width=int(poly_Height*actual_width/actual_height)
        
        polygons_Parameterization[f'poly{str(poly_idx)}']={
            "origin":tuple([x_min,y_min]),
            "center":tuple(center),
            'poly_ActualSize':(actual_width,actual_height),
            'poly_ImageSize':(poly_Width,poly_Height),
            'basis':tuple([list(x_axis),list(y_axis),list(z_axis)]),
            "faces_idx":tuple(current_face_Set),
            "edges_idx":tuple(current_edge_Set)
        }
        poly_idx+=1
        

    print("polygons_Parameterization:",polygons_Parameterization)
    texture_image = get_texture_image_by_object(obj)
    ensure_directory_exists(os.path.join(output_path,obj_name,"images"))
    if texture_image:
        for key,value in polygons_Parameterization.items():
            # save_texture_for_faces(mesh, value['faces_idx'], texture_image, f"{key}.png")
            FacadeTexture=generate_FacadeTexture(mesh,texture_image,value)
            print(f"当前将保存{key}")
            polyImageName=os.path.join(output_path,obj_name,"images",f"{key}.jpg")
            cv2.imwrite(polyImageName, FacadeTexture)
    json_path=os.path.join(output_path,obj_name,"data.json")
    with open(json_path, "w") as json_file:
        json.dump(polygons_Parameterization, json_file, indent=4)






main()