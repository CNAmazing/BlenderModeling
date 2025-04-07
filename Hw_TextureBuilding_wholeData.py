import numpy as np
import bpy
import mathutils
import os
import cv2
import sys
cwd = os.getcwd()
sys.path.append(cwd)
from Hw_Tools import *


      
def main():

    folder_path=r'output\001'
    total_classes=['window','door','glass']
    json_path=os.path.join(folder_path,'data.json')
    data=read_json(json_path)
    # print(f'data:{data}')
    """
    data格式
    key:poly6
    value:{
    'origin': [2.099999741096366, -2.8875637054708485], 
    'center': [2.862616777420044, 0.6875329613685608, -2.332228899002075], 
    'poly_ActualSize': [1.129995043940347, 1.6660039425336588], ————————>宽，高表示
    'poly_ImageSize': [434, 640], ————————>宽，高表示
    'basis': [[0.8848744556019921, 0.4658295802362471, -0.0], [0.0, 0.0, 1.0], [0.4658295214176178, -0.8848743438720703, 0.0005235497374087572]], 
    'faces_idx': [11, 21], 
    'edges_idx': [[12, 13], [7, 13], [5, 7], [5, 13], [7, 12]]}
    'window':[[]]
    'glass':[[]]    
    'door'[[]]
    'windowDepth': 0.0,
    'glassDepth': 0.0,
    'doorDepth': 0.0,
    'facadeDepth': 0.0,
    """
    polygonPlaneList=[]
    count_classes=[]
    name = os.path.basename(folder_path)
    obj = bpy.context.scene.objects.get(name)  
    if obj.type != 'MESH':
        raise ValueError(f"{obj.name} 不是网格对象")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for poly_idx,poly_info in data.items():
        print(f'正在处理多边形面片:{poly_idx}')
        image_path=os.path.join(folder_path,'images',poly_idx+'.jpg')
        normalImage_path=os.path.join(folder_path,'images',poly_idx+'_normal.jpg')
        depthImage_path=os.path.join(folder_path,'images',poly_idx+'_depth.jpg')
        current_classes=[cls for cls in total_classes if cls in poly_info]
        # for cls in current_classes:
        #     poly_info[cls]=xyxy_to_xywh(poly_info[cls])

            
        #blender读取图片顺序是从左下角开始，而numpy读取图片是从左上角开始
        image = cv2.imread(image_path)
        image_height,image_width,_=image.shape
        depthImage=cv2.imread(depthImage_path)
        depth_Dict={}
        for cls in total_classes:
            if cls in current_classes:
                depth_Dict[cls]=create_boolean_array(image_height,image_width,poly_info[cls])
            else:
                depth_Dict[cls]=np.zeros((image_height, image_width), dtype=bool)
        depth_Dict['facade']=np.ones((image_height, image_width), dtype=bool)
        for key,value in depth_Dict.items():
            match key:
                case 'facade':
                    depth_Dict[key]=depth_Dict[key] & ~depth_Dict['window'] & ~depth_Dict['door']
                case 'window':
                    depth_Dict[key]=depth_Dict[key] & ~depth_Dict['glass'] 

        poly_info['facadeDepth']=np.mean(depthImage[depth_Dict['facade']])
        threshold=10
        for cls in current_classes:
            poly_info[f'{cls}Depth']=np.mean(depthImage[depth_Dict[cls]])-poly_info['facadeDepth']
            poly_info[f'{cls}Depth']=poly_info[f'{cls}Depth']/threshold
        if 'glass' in current_classes:
            poly_info['glassDepth']=poly_info['glassDepth']-poly_info['windowDepth']
        print(f'poly_info:{poly_info}')
        """
        主程序阶段
        """
        for c in current_classes:
            if c not in count_classes:
                count_classes.append(c)
        R=np.array(poly_info['basis'])
        R=R.T
        T=np.array(poly_info['center'])
        R_before=np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        R_after=np.array(poly_info['basis'])
        rotation_matrix = np.dot(R_before, np.linalg.inv(R_after))
        euler = mathutils.Matrix(rotation_matrix).to_euler('XYZ')
        # vertices = np.array([np.array(obj.data.vertices[idx].co) for idx in poly.vertices])
        actual_width = poly_info['poly_ActualSize'][0]
        actual_height= poly_info['poly_ActualSize'][1]
        polygonPlane = PolygonPlane(np.array(poly_info['basis'][2]), np.array(poly_info['center']))
        for cls in current_classes:
            setattr(polygonPlane,f"{cls}_depth",poly_info[f"{cls}Depth"])
                
        
        polygonPlane.R=R
        polygonPlane.actual_width=actual_width
        polygonPlane.actual_height=actual_height
        polygonPlane.classes=current_classes
        polygonPlaneList.append(polygonPlane)
        generate_cube_by_currentClasses_PolyInfo(current_classes,
                                        poly_info,
                                        euler)
    
    for cls in count_classes:
        c_collection = bpy.data.collections[f"{cls}Collection"]
        apply_boolean(obj,c_collection)
    
    print('布尔运算完成！')
        
    
    # facade_poly_idxs=[]
    window_poly_idxs=[]
    glass_poly_idxs=[]
    door_poly_idxs=[]

    for polygonPlane in polygonPlaneList:
    # 获取物体的变换矩阵
        for poly in obj.data.polygons:
            vertices = [obj.data.vertices[idx].co for idx in poly.vertices]
            # 判断面片是否在平面上
            # if all(is_point_on_plane(point, polygonPlane.facade_plane_equation) for point in vertices):
            #     facade_poly_idxs.append(poly.index)

            for cls in polygonPlane.classes:
                c_plane_equation=getattr(polygonPlane,f"{cls}_plane_equation")
                c_poly_idxs=locals()[f"{cls}_poly_idxs"]
                if all(is_point_on_plane(point, c_plane_equation) for point in vertices):
                    c_poly_idxs.append(poly.index)
            

    """
    调整UV
    """
    
    # uv_layer_name = "UVMap"  
    # mesh = obj.data
    # uv_layer = mesh.uv_layers[uv_layer_name]
    # mesh.uv_layers.active = uv_layer

    # for polygonPlane in polygonPlaneList:
    #     for cls in polygonPlane.classes:
    #         c_poly_idxs=locals()[f"{cls}_poly_idxs"]
    #         c_plane_equation=getattr(polygonPlane,f"{cls}_plane_equation")
    #         for face_index in c_poly_idxs:
    #             face = mesh.polygons[face_index]

    #             vertex_coords = np.array([np.array(mesh.vertices[vertex_idx].co) for vertex_idx in face.vertices])
    #             if all(is_point_on_plane(vertex, c_plane_equation) for vertex in vertex_coords):
    #                 for loop_index,vertex in zip(face.loop_indices,vertex_coords):
    #                     # 获取 UV 坐标
    #                     uv = uv_layer.data[loop_index].uv
    #                     u,v=polygonPlane.point_to_UV(vertex)
    #                     # 调整 UV 坐标（示例：简单映射）
    #                     uv[0] = u  # U 坐标
    #                     uv[1] = v

    #             # vertices = np.array([np.array(loop.vert.co) for loop in face.loops])
    #                 # ajust_UV(bm_uv_layer,face,polygonPlane)
    # bpy.ops.object.mode_set(mode='OBJECT')
    
    # #连接法线贴图 并修改粗糙度
    # base_texture=obj.material_slots[0].material
    # normal_texture = base_texture.node_tree.nodes.new(type='ShaderNodeTexImage')
    # normal_texture.image = bpy.data.images.load(normalImage_path)
    # normal_principled_bsdf = base_texture.node_tree.nodes.get("原理化 BSDF")
    # normal_principled_bsdf.inputs['Roughness'].default_value =1.0
    # base_texture.node_tree.links.new(normal_texture.outputs['Color'], normal_principled_bsdf.inputs['Normal'])

    # #创建玻璃材质 
    # image_blender=bpy.data.images.load(image_path)
    # material=bpy.data.materials.new("glassMaterial")  
    # material.use_nodes = True
    # # 获取材质的节点树
    # nodes = material.node_tree.nodes
    # # 创建图像纹理节点
    # image_texture = nodes.new(type='ShaderNodeTexImage')
    # # 加载图像
    # image_texture.image = image_blender
    # # 获取纹理坐标节点
    # texture_coord = nodes.new(type='ShaderNodeTexCoord')
    # # 获取Principled BSDF节点
    # principled_bsdf = nodes.get("原理化 BSDF")
    # # 连接纹理坐标到图像纹理节点
    # material.node_tree.links.new(texture_coord.outputs['UV'], image_texture.inputs['Vector'])
    # # 连接图像纹理到 Principled BSDF 的 Base Color
    # material.node_tree.links.new(image_texture.outputs['Color'], principled_bsdf.inputs['Base Color'])    
    # principled_bsdf.inputs['Metallic'].default_value = 0.0 # 设置金属度，0.0为非金属，1.0为金属 
    # principled_bsdf.inputs['Roughness'].default_value = 0.0  # 设置粗糙度，0.0为光滑，1.0为粗糙
    # # 将材质应用到选中的物体
    # obj.data.materials.append(material)
    # for face_index in glass_poly_idxs:
    #     add_material_by_faceIdx(obj,face_index,"glassMaterial")    

main()