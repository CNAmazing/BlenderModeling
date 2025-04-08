import numpy as np
import bpy
import mathutils
import sys
import os
import cv2

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
        depthImage_path=os.path.join(folder_path,'images',poly_idx+'_depth.tiff')
        current_classes=[cls for cls in total_classes if cls in poly_info]
        
            
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
        
    
    facade_poly_idxs=[]
    window_poly_idxs=[]
    glass_poly_idxs=[]
    door_poly_idxs=[]

    for polygonPlane in polygonPlaneList:
    # 获取物体的变换矩阵
        for poly in obj.data.polygons:
            vertices = [obj.data.vertices[idx].co for idx in poly.vertices]
            # 判断面片是否在平面上
            if all(is_point_on_plane(point, polygonPlane.facade_plane_equation) for point in vertices):
                facade_poly_idxs.append(poly.index)

            for cls in polygonPlane.classes:
                c_plane_equation=getattr(polygonPlane,f"{cls}_plane_equation")
                c_poly_idxs=locals()[f"{cls}_poly_idxs"]
                if all(is_point_on_plane(point, c_plane_equation) for point in vertices):
                    c_poly_idxs.append(poly.index)
            
  

    color_dict={'facade':(69,5,89),
             'window':(255,0,0),
             'glass':(0,0,255),
             'door':(250,86,53),
             'balcony':(32,140,62),}
    print(f"facade_poly_idxs:{facade_poly_idxs}")
    print(f"window_poly_idxs:{window_poly_idxs}")
    print(f"glass_poly_idxs:{glass_poly_idxs}")
    print(f"door_poly_idxs:{door_poly_idxs}")
    create_material(color_dict)
    add_material_to_object(obj,color_dict)
    for face_index in facade_poly_idxs:
        add_material_by_faceIdx(obj,face_index,"facade")
    if 'window' in count_classes:
        for face_index in window_poly_idxs:
            add_material_by_faceIdx(obj,face_index,"window")
    if 'glass' in count_classes:
        for face_index in glass_poly_idxs:
            add_material_by_faceIdx(obj,face_index,"glass")
    if 'door' in count_classes:
        for face_index in door_poly_idxs:
            add_material_by_faceIdx(obj,face_index,"door")   

        
main()