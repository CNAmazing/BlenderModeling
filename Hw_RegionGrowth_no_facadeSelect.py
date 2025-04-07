import bpy
import sys
import os
import random
cwd = os.getcwd()
sys.path.append(cwd)
from Hw_Tools import *

def main():
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

    total_face_Set=set() 
    for face in mesh.polygons:
        # 生成随机颜色
        current_face_Set=set()
        current_edge_Set=set()
        normal=face.normal
        print(normal)
        # if not face_select_by_normal(normal) :
        #     continue
        if face.index in total_face_Set:
            continue
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
        color = (*(
    [random.uniform(0.9, 1.0), random.uniform(0.0, 0.1), random.uniform(0.0, 1.0)][i] 
    for i in random.sample([0, 1, 2], k=3)
), 1.0)
        material = bpy.data.materials.new(name=f"FaceColor_{face.index}")
        material.diffuse_color = color
        mesh.materials.append(material)
        for c_face in current_face_Set:
            mesh.polygons[c_face].material_index = len(mesh.materials) - 1
            
    print(total_face_Set)

main()