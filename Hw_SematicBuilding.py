import numpy as np
import bpy
import mathutils
import bmesh
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
from Hw_Tools import *

def main():
    xyxy_window= np.array([
        [88, 91, 146, 168],
        [233, 91, 292, 168],
        [380, 91, 437, 168],
        [88,  246, 146, 336],
        [233, 246, 292, 336],
        [380, 246, 437, 336],
        [73, 421, 156, 521],
        [220, 421, 306, 521],
        [366, 421, 451, 521],
        
                          
                ])
    xyxy_glass=np.array([
                         [93, 101, 145, 120],
                         [236, 101, 288, 120],
                         [381, 101, 433, 120],
                         [94, 130,115, 169],
                         [122, 130, 144,168],
                         [236,130, 259, 168],
                         [265, 130, 288, 168],
                         [381, 130, 405, 168],
                         [410, 130, 433, 168],

                         [93, 257, 144, 280],
                         [236, 257, 288, 280],
                         [381, 257, 433, 280],

                         [92, 287, 116, 336],
                         [121, 287, 143, 336],
                         [236, 287, 260, 336],
                         [266, 287, 290, 336],
                         [381, 287, 404, 336],
                         [411, 287, 433, 336],

                         [79, 428, 156, 454],
                         [224, 428, 303, 454],
                         [368, 428, 445, 454],
                         [81, 462, 116, 521],
                         [121, 462, 157, 521],
                         [225, 462, 260, 521],
                         [266, 462, 303, 521],
                         [370, 462, 404, 521],
                         [411, 462, 445, 521],

                         
                ])
    
    xywh_window= xyxy_to_xywh(xyxy_window)
    xywh_glass= xyxy_to_xywh(xyxy_glass)
    # xywh_door= xyxy_to_xywh(xyxy_door)
    image_path = r"E:\Desktop\facade\00259_.jpg"
    image = bpy.data.images.load(image_path)
    image_width, image_height = image.size[0],image.size[1]
    # depth_image_path = r"E:\Desktop\Simulationtest\Simulationtest\depth.png"
    # depth_image = bpy.data.images.load(depth_image_path)

    window_bool_array = create_boolean_array(image_width, image_height, xywh_window)
    glass_bool_array = create_boolean_array(image_width, image_height, xywh_glass)
    facade_bool_array = np.ones((image_height, image_width), dtype=bool)

    facade_bool_array=facade_bool_array & ~window_bool_array
    window_bool_array=window_bool_array & ~glass_bool_array
    """
    blender读取图片顺序是从左下角开始，而numpy读取图片是从左上角开始
    """

    # pixels = np.array(depth_image.pixels[:])
    # pixels=pixels[::4]
    # pixels=pixels.reshape(-1,image_width)
    # pixels=np.flip(pixels,axis=0)
    
    # facade_depth = np.mean(pixels[facade_bool_array])
    # window_depth = np.mean(pixels[window_bool_array])
    # glass_depth = np.mean(pixels[glass_bool_array])
    # window_depth=facade_depth-window_depth
    # glass_depth=facade_depth-window_depth-glass_depth
    # print(f"facade_depth: {facade_depth}")
    # print(f"window_depth: {window_depth}")
    # print(f"glass_depth: {glass_depth}")

    facade_parameterization={
        "2":{
            'window':xywh_window,
            'glass':xywh_glass,
            'windowDepth':0.1,
            'glassDepth':0.2,
            'doorDepth':0.7,
            },
        "3":{
            'window':xywh_window,
            'glass':xywh_glass,
            'windowDepth':0.1,
            'glassDepth':0.2,
            'doorDepth':0.7,
            },
        "4":{
            'window':xywh_window,
            'glass':xywh_glass,
            'windowDepth':0.1,
            'glassDepth':0.2,
            'doorDepth':0.7,

            },
        "5":{
            'window':xywh_window,
            'glass':xywh_glass,
            'windowDepth':0.1,
            'glassDepth':0.2,
            'doorDepth':0.7,
            },
    }
    obj = bpy.context.scene.objects.get("Cube")  

    if obj.type != 'MESH':
        raise ValueError(f"{obj.name} 不是网格对象")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    polygonPlaneList=[]
    total_classes=['window','door','glass']
    count_classes=[]
    for poly in obj.data.polygons:
        normal = poly.normal
        center = poly.center
        poly_index=str(poly.index)
        print(f'poly_index:{poly_index},normal:{normal}')
        if poly_index not in facade_parameterization:
            continue
        current_classes=[key for key in total_classes if key in facade_parameterization[poly_index]]
        for c in current_classes:
            if c not in count_classes:
                count_classes.append(c)
        normal =np.array(normal)
        center = np.array(center)
        y_axis=np.array([0,0,1])
        x_axis=np.cross(y_axis,normal)

        x_axis=x_axis/np.linalg.norm(x_axis)
        y_axis=y_axis/np.linalg.norm(y_axis)
        z_axis = normal/np.linalg.norm(normal)
        
        R=np.array([x_axis,y_axis,z_axis])
        R=R.T
        T=center
        R_before=np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
        R_after=np.array([x_axis, y_axis, z_axis])
        rotation_matrix = np.dot(R_before, np.linalg.inv(R_after))
        euler = mathutils.Matrix(rotation_matrix).to_euler('XYZ')
       
        vertices = np.array([np.array(obj.data.vertices[idx].co) for idx in poly.vertices])
        actual_width = max(np.dot(vertices, x_axis)) - min(np.dot(vertices, x_axis))
        actual_height = max(np.dot(vertices, y_axis)) - min(np.dot(vertices, y_axis))
        

        polygonPlane = PolygonPlane(normal, center)
        for cls in current_classes:
                setattr(polygonPlane,f"{cls}_depth",facade_parameterization[poly_index][f"{cls}Depth"])
        polygonPlane.R=R
        polygonPlane.actual_width=actual_width
        polygonPlane.actual_height=actual_height
        polygonPlaneList.append(polygonPlane)
        generate_cube_by_currentClasses(current_classes,
                                        facade_parameterization,
                                        poly_index,
                                        R,T,
                                        image_width,
                                        image_height,
                                        actual_width,
                                        actual_height,
                                        euler)

    for cls in count_classes:
            c_collection = bpy.data.collections[f"{cls}Collection"]
            apply_boolean(obj,c_collection)

        
    
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
            
            for cls in count_classes:
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