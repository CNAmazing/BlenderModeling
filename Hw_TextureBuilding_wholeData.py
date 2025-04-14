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
    obj_name = '003_LOD'
    folder_path=os.path.join('output',obj_name)
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
        depthImage_path=os.path.join(folder_path,'images',poly_idx+'_depth.tiff')
        roughnessImage_path=os.path.join(folder_path,'images',poly_idx+'_roughness.jpg')
        current_classes=[cls for cls in total_classes if cls in poly_info]
        # for cls in current_classes:
        #     poly_info[cls]=xyxy_to_xywh(poly_info[cls])

        mask=poly_info['mask']    
        #blender读取图片顺序是从左下角开始，而numpy读取图片是从左上角开始
        image = cv2.imread(image_path)
        image_height,image_width,_=image.shape
        depthImage=cv2.imread(depthImage_path,cv2.IMREAD_UNCHANGED)
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
                    depth_Dict[key]=mask&depth_Dict[key] & ~depth_Dict['window'] & ~depth_Dict['door']
                case 'window':
                    depth_Dict[key]=mask&depth_Dict[key] & ~depth_Dict['glass'] 
                case 'glass':
                    roughnessMap = np.where(~depth_Dict[key], 255, 0).astype(np.uint8)
                    # 保存图像
                    cv2.imwrite(roughnessImage_path, roughnessMap)
        threshold=10
        poly_info['facadeDepth']=np.mean(depthImage[depth_Dict['facade']])/threshold
        
        for cls in current_classes:
            poly_info[f'{cls}Depth']=np.mean(depthImage[depth_Dict[cls]])/threshold
        print('优化前数据')
        for key in poly_info.keys():
            if key.endswith('Depth'):
                print(f'{key}:{poly_info[key]}')
        for cls in current_classes:    
            poly_info[f'{cls}Depth']=poly_info[f'{cls}Depth']-poly_info['facadeDepth']
            
            
        if 'glass' in current_classes:
            poly_info['glassDepth']=poly_info['glassDepth']-poly_info['windowDepth']
        filtered_info = {k: v for k, v in poly_info.items() if k != 'mask'}
        print(f'poly_info:{filtered_info}')
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
        polygonPlane.classes=current_classes+['facade']
        polygonPlane.poly_idx=poly_idx
        polygonPlaneList.append(polygonPlane)
        generate_cube_by_currentClasses_PolyInfo(current_classes,
                                        poly_info,
                                        euler)
    
    for cls in count_classes:
        c_collection = bpy.data.collections[f"{cls}Collection"]
        apply_boolean(obj,c_collection)
    
    print('布尔运算完成！')
        
    
    # facade_poly_idxs=[]
    # window_poly_idxs=[]
    # glass_poly_idxs=[]
    # door_poly_idxs=[]

    for polygonPlane in polygonPlaneList:
    # 获取物体的变换矩阵
        poly_idx=polygonPlane.poly_idx
        for cls in polygonPlane.classes:
            c_plane_equation=getattr(polygonPlane,f"{cls}_plane_equation")
            for poly in obj.data.polygons:
                vertices = np.array([np.array(obj.data.vertices[idx].co) for idx in poly.vertices])
                # c_poly_idxs=locals()[f"{cls}_poly_idxs"]
                if all(is_point_on_plane(point, c_plane_equation) for point in vertices):
                    # c_poly_idxs.append(poly.index)
                    data[poly_idx].setdefault(f"{cls}_poly_idxs",[]).append(poly.index)

    """
    调整UV
    """
    uv_layer_name = "UVMap"  
    mesh = obj.data

    # if uv_layer_name not in mesh.uv_layers:
    #     print(f"错误：UV 层 '{uv_layer_name}' 不存在！")
    #     print(f"可用的 UV 层：{[layer.name for layer in mesh.uv_layers]}")
    #     # 可以选择创建一个新的 UV 层或使用默认层
    #     uv_layer_name = mesh.uv_layers.new(name=uv_layer_name).name
    uv_layer = mesh.uv_layers[uv_layer_name]
    mesh.uv_layers.active = uv_layer

    for polygonPlane in polygonPlaneList:

        for cls in polygonPlane.classes:
            try:
                c_poly_idxs=data[polygonPlane.poly_idx][f"{cls}_poly_idxs"]
            except KeyError:
                continue
            for face_index in c_poly_idxs:
                face = mesh.polygons[face_index]

                vertex_coords = np.array([np.array(mesh.vertices[vertex_idx].co) for vertex_idx in face.vertices])

                for loop_index,vertex in zip(face.loop_indices,vertex_coords):
                    # 获取 UV 坐标
                    uv = uv_layer.data[loop_index].uv
                    u,v=polygonPlane.point_to_UV(vertex)
                    # 调整 UV 坐标（示例：简单映射）
                    uv[0] = u  # U 坐标
                    uv[1] = v

                
    bpy.ops.object.mode_set(mode='OBJECT')
    
    for polygonPlane in polygonPlaneList:
        poly_idx=polygonPlane.poly_idx
        path=r'E:\hhhh\BlenderModeling\output\003_LOD'
        image_path=os.path.join(path,'images',poly_idx+'.jpg')
        normalImage_path=os.path.join(path,'images',poly_idx+'_normal.jpg')
        roughnessImage_path=os.path.join(path,'images',poly_idx+'_roughness.jpg')
        # image_path='output'+'/'+obj_name+'/'+'images'+'/'+poly_idx+'.jpg'
        # normalImage_path='output'+'/'+obj_name+'/'+'images'+'/'+poly_idx+'_normal.jpg'
        material=bpy.data.materials.new(f"{poly_idx}")  
        material.use_nodes = True
        # 创建图像纹理节点
        image_texture = material.node_tree.nodes.new(type='ShaderNodeTexImage')
        # 加载图像
        image_texture.image = bpy.data.images.load(image_path)

        normalImage_texture=material.node_tree.nodes.new(type='ShaderNodeTexImage')
        normalImage_texture.image=bpy.data.images.load(normalImage_path)
        roughnessImage_texture=material.node_tree.nodes.new(type='ShaderNodeTexImage')
        roughnessImage_texture.image=bpy.data.images.load(roughnessImage_path)
        # 获取纹理坐标节点
        texture_coord = material.node_tree.nodes.new(type='ShaderNodeTexCoord')
        # 获取Principled BSDF节点
        principled_bsdf = material.node_tree.nodes.get("原理化 BSDF")
        # 连接纹理坐标到图像纹理节点
        material.node_tree.links.new(texture_coord.outputs['UV'], image_texture.inputs['Vector'])
        material.node_tree.links.new(texture_coord.outputs['UV'], normalImage_texture.inputs['Vector'])
        material.node_tree.links.new(texture_coord.outputs['UV'], roughnessImage_texture.inputs['Vector'])
        # 连接图像纹理到 Principled BSDF 的 Base Color
        material.node_tree.links.new(image_texture.outputs['Color'], principled_bsdf.inputs['Base Color'])    
        # material.node_tree.links.new(normalImage_texture.outputs['Color'], principled_bsdf.inputs['Normal'])
        material.node_tree.links.new(roughnessImage_texture.outputs['Color'], principled_bsdf.inputs['Roughness'])
        principled_bsdf.inputs['Metallic'].default_value =0.5 # 设置金属度，0.0为非金属，1.0为金属 
        # principled_bsdf.inputs['Roughness'].default_value = 1.0  # 设置粗糙度，0.0为光滑，1.0为粗糙
        blenderNode=material.node_tree.nodes.new(type='ShaderNodeMix')
        blenderNode.data_type = 'RGBA'
        material.node_tree.links.new(roughnessImage_texture.outputs['Color'], blenderNode.inputs[0])
        blenderNode.inputs[6].default_value=(0.5, 0.5, 1.0, 1.0) # 设置颜色
        material.node_tree.links.new(normalImage_texture.outputs['Color'], blenderNode.inputs[7])
        output_node =  material.node_tree.nodes.get('材质输出')
        material.node_tree.links.new(blenderNode.outputs['Result'], output_node.inputs['Displacement'])
        
        obj.data.materials.append(material)
        
        for cls in polygonPlane.classes:
            try:
                c_poly_idxs=data[poly_idx][f"{cls}_poly_idxs"]
            except KeyError:
                continue
            for face_index in c_poly_idxs:
                add_material_by_faceIdx(obj,face_index,poly_idx)
    mesh.uv_layers[uv_layer_name].active_render = True
   
    print('Done!')
main()