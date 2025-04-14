import bpy
import sys
import os
import random
from mathutils import Vector
cwd = os.getcwd()
sys.path.append(cwd)
from Hw_Tools import *

def main():
    
    obj_name='Cube'
    obj = bpy.data.objects.get(obj_name)

    if obj is None:
        raise ValueError(f"未找到名为 {obj_name} 的物体")

    # 确保物体是网格类型
    if obj.type != 'MESH':
        raise ValueError("物体不是网格类型")

    mesh = obj.data

    # 确保物体有材质
    if not mesh.materials:
        material = bpy.data.materials.new(name="RandomColor")
        mesh.materials.append(material)

    
    total_face_Set=set() 
    polygonPlane_List=[]
    polyCount=0
    for face in mesh.polygons:
        # 生成随机颜色
        current_face_Set=set()
        current_edge_Set=set()
        normal=face.normal
        
        print(normal)
        if not face_select_by_normal(normal) :
            continue
        if face.index in total_face_Set:
            continue
        polygonPlane=PolygonPlane(face.normal,face.center)
        polygonPlane_List.append(polygonPlane)
        current_face_Set.add(face.index)
        total_face_Set.add(face.index)
        for edge in face.edge_keys:
            current_edge_Set.add(edge)
        last_len=len(current_face_Set)
        current_len=0
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
        #创建新材质
        color = (*((1.0, 0.0, random.random())[i] for i in random.sample([0, 1, 2], k=3)), 1.0)
        material = bpy.data.materials.new(name=f"FaceColor_{face.index}")
        material.diffuse_color = color
        mesh.materials.append(material)
        for c_face in current_face_Set:
            mesh.polygons[c_face].material_index = len(mesh.materials) - 1
        polyCount+=1
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='FACE')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.separate(type='MATERIAL')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    move_distance = 1.5  
    world_origin = obj.matrix_world.translation
    print(f'world_origin:{world_origin}')
    for i in range(polyCount):
        idx=i+1
        poly_name=f'{obj_name}.{idx:03d}'
        print(poly_name)
        poly_obj=bpy.data.objects.get(poly_name)
        first_poly = poly_obj.data.polygons[0]
        normal = first_poly.normal
        v = first_poly.center  - world_origin
        # 计算点积
        dot_product = v.dot(normal)
        # 判断是否在法向半球面内
        if dot_product < 0:
            normal = -normal

        # 沿法向移动物体
        poly_obj.location += normal.normalized() * move_distance
    
    poly_name=obj_name
    poly_obj=bpy.data.objects.get(poly_name)
    first_poly = poly_obj.data.polygons[0]
    normal = first_poly.normal
    v = first_poly.center  - world_origin
    # 计算点积
    dot_product = v.dot(normal)
    # 判断是否在法向半球面内
    if dot_product > 0:
        normal = -normal
    # 沿法向移动物体
    poly_obj.location += normal.normalized() * move_distance
    
    
    print(total_face_Set)
    print(f'共有{polyCount}个立面多边形')

main()