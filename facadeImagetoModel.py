import bpy
import cv2
import numpy as np
import math
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
from Hw_Tools import *
def get_jpg_paths(folder_name):
    """
    获取指定文件夹下images子目录中的所有.jpg图片路径及不带后缀的文件名
    
    参数:
        folder_name (str): 目标文件夹名称（如"x"）
        
    返回:
        tuple: (完整路径列表, 不带后缀的文件名列表)，如(
                ["x/images/pic1.jpg", "x/images/pic2.jpg"], 
                ["pic1", "pic2"]
               )
    """
    full_paths = []
    basenames = []
    
    try:
        images_dir = folder_name
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"目录不存在: {images_dir}")
            
        for f in os.listdir(images_dir):
            if f.lower().endswith(".jpg"):
                file_path = os.path.join(images_dir, f)
                if os.path.isfile(file_path):
                    full_paths.append(file_path)
                    # 去掉.jpg后缀获取纯文件名
                    basenames.append(os.path.splitext(f)[0])
                    
        return full_paths, basenames
        
    except Exception as e:
        print(f"错误: {e}")
        return [], []
def add_material_by_faceIdx(obj,faceIdx,material_name):
    """
    为物体的指定面添加材质
    :param object: 物体
    :param faceIdx: 面的索引
    :param material_name: 材质名称
    """
    # 获取网格数据
    mesh = obj.data
    # 获取材质索引
    material_index = obj.data.materials.find(material_name)
    # 获取指定面片（例如索引为0的面片）
    mesh.polygons[faceIdx].material_index = material_index
    # 切换回物体模式
    # bpy.ops.object.mode_set(mode='OBJECT')
def face_select_by_normal(v1,v2=(0,0,1),tolerance_degrees=1):
    """
    判断两个向量是否正交。
    :param v1: 第一个向量，例如 (x1, y1, z1)
    :param v2: 第二个向量，例如 (x2, y2, z2)
    :param tolerance_degrees: 角度容差（默认1度）
    :return: 是否正交（True/False）
    """
    # 计算点积
    dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    
    # 计算向量的模
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    
    # 计算夹角的余弦值
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    
    # 避免浮点数精度问题导致cos_angle超出[-1, 1]范围
    cos_angle = max(min(cos_angle, 1), -1)
    
    # 计算夹角（弧度）
    angle_rad = math.acos(cos_angle)
    
    # 将夹角转换为角度
    angle_deg = math.degrees(angle_rad)
    
    # 判断是否正交
    return abs(angle_deg - 90) < tolerance_degrees
def image_to_model(image_path):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 创建一个正方体
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    basename=get_basename_without_suffix(image_path)
    image=cv2.imread(image_path)
    height,width,channal=image.shape

    # 获取当前选中的对象（即刚刚创建的正方体）
    cube = bpy.context.object
    cube.name = basename
    # 设置 XYZ 缩放
    cube.scale = (width/2/10, width/2/10, height/2/10)  # 设置 X, Y, Z 缩放比例

    # 应用缩放变换
    bpy.ops.object.transform_apply(scale=True)

    obj = bpy.data.objects.get(basename)
    mesh = obj.data
    UV_Coordinates=(1,0),(1,1),(0,1),(0,0)
    uv_layer_name = "UVMap"
    image_bl=bpy.data.images.load(image_path) 

    if "white" not in bpy.data.materials:
        material = bpy.data.materials.new(name="white")
        material.diffuse_color = (1, 1, 1, 1)
        mesh.materials.append(material)
    else :
        material=bpy.data.materials.get("white")
        mesh.materials.append(material)

    if "MyMaterial" not in bpy.data.textures:
        bpy.ops.object.mode_set(mode='OBJECT')
        material = bpy.data.materials.new(name="MyMaterial")

        # 创建纹理
        # texture = bpy.data.textures.new(name="MyTexture", type='IMAGE')
        # texture.image = bpy.data.images.load(image_path)  # 替换为你的纹理路径
        
        # 将纹理添加到材质
        material.use_nodes = True
        nodes = material.node_tree.nodes
        
        principled_bsdf = nodes.get("原理化 BSDF")
        texture_coord = nodes.new(type='ShaderNodeTexCoord')
        tex_image = nodes.new('ShaderNodeTexImage')
        tex_image.image = image_bl
        material.node_tree.links.new(texture_coord.outputs['UV'], tex_image.inputs['Vector'])
        material.node_tree.links.new( tex_image.outputs['Color'],principled_bsdf.inputs['Base Color'])
        mesh.materials.append(material) 

    for poly in mesh.polygons:
        normal=poly.normal
        if not face_select_by_normal(normal):
            add_material_by_faceIdx(obj,poly.index,"white")
            continue
        
        if uv_layer_name in mesh.uv_layers:
            uv_layer = mesh.uv_layers[uv_layer_name]
            mesh.uv_layers.active = uv_layer
        else:
            # 如果不存在，创建新的 UV 层并激活
            uv_layer = mesh.uv_layers.new(name=uv_layer_name)
            mesh.uv_layers.active = uv_layer

        for loop_index,UV_s in zip(poly.loop_indices,UV_Coordinates):
                # 获取 UV 坐标
                uv = uv_layer.data[loop_index].uv
                # 调整 UV 坐标（示例：简单映射）
                uv.x = UV_s[0]  # U 坐标
                uv.y = UV_s[1]  # V 坐标

        add_material_by_faceIdx(obj,poly.index,"MyMaterial")
    mesh.uv_layers['UVMap'].active_render = True
def export_obj(obj_name, export_path):
        # 确保物体存在
    if obj_name not in bpy.data.objects:
        raise Exception(f"名为 {obj_name} 的物体不存在")

    # 设置导出路径（默认保存到 blend 文件所在目录）

    # 选中目标物体（取消其他选择）
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[obj_name].select_set(True)

   
    bpy.ops.wm.obj_export(
        filepath=export_path,
        forward_axis='X',
        up_axis='Z',
        export_selected_objects=True,
        )
    print(f"成功导出到: {export_path}")
def clear_all_materials():
    """清除 Blender 中所有材质"""
    for material in list(bpy.data.materials):
          # 使用 list() 避免修改集合时的迭代问题
            bpy.data.materials.remove(material)
def main():
    input_folder = r"E:\Desktop\facadeData"
    jps_paths,basenames=get_jpg_paths(input_folder)
    for jpg_path,basename in zip(jps_paths,basenames):
        clear_all_materials()
        image_to_model(jpg_path)
        export_path = os.path.join(input_folder, f"{basename}.obj")
        export_obj(basename, export_path)
    # clear_all_materials()
    # image_path=r"E:\Desktop\facadeData\00472.jpg"
    # basename=image_to_model(image_path)
    # export_path = bpy.path.abspath(f"E:/Desktop/facadeData/{basename}.obj")
    # export_obj(basename, export_path)
    
main()