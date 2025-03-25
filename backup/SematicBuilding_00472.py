import numpy as np
import bpy
import mathutils
import bmesh
def xyxy_to_xywh(bboxes):
    array=[]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        array.append([x1, y1, x2 - x1, y2 - y1])
    return np.array(array)
class PolygonPlane():
    def __init__(self,normal, point):
        self.original_point = point
        self.normal = normal
        self.window_depth=None
        self.glass_depth=None
        self.door_depth=None
        self.R=None
        self.actual_width=None
        self.actual_height=None
    @property
    def facade_plane_equation(self):
        return self.plane_equation(self.normal, self.original_point)
    @property
    def window_plane_equation(self):
        if self.window_depth is None:
            raise ValueError("window_depth is None")
        return self.plane_equation(self.normal, self.original_point-self.window_depth*self.normal)
    @property
    def door_plane_equation(self):
        if self.door_depth is None:
            raise ValueError("door_depth is None")
        return self.plane_equation(self.normal, self.original_point-self.door_depth*self.normal)
    @property
    def glass_plane_equation(self):
        if self.glass_depth is None:
            raise ValueError("glass_depth is None")
        return self.plane_equation(self.normal, self.original_point-self.glass_depth*self.normal-self.window_depth*self.normal)
    def point_to_UV(self,point):
        
        """
        R已经转置 每一列是一个基向量
        """
        p0=np.dot(point-self.original_point,self.R)
        p0[0]+=self.actual_width/2
        p0[1]+=self.actual_height/2
        u = p0[0] / self.actual_width
        v = p0[1] / self.actual_height
        return u, v
    def plane_equation(self,normal_vector, point):
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
    return abs(A * x + B * y + C * z + D )<=0.0001
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
        if material_name not in bpy.data.materials:
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
    # 获取网格数据
    mesh = obj.data
    # 切换到对象模式以访问面片数据
    # 获取材质索引
    material_index = obj.data.materials.find(material_name)
    # 获取指定面片（例如索引为0的面片）
    mesh.polygons[faceIdx].material_index = material_index
    # 切换回物体模式
 # 临时禁用 print
def get_locations(xywh,R,T):
    #将矩形的左下角坐标转换为中心坐标
    
    center_coordinates = xywh[:, :2] + xywh[:, 2:] / 2
    points_3d = np.column_stack((center_coordinates, np.zeros(center_coordinates.shape[0])))
    # points_3d[:, [0, 1]] = points_3d[:, [1, 0]]
    #正常情况R*p  p为单个点 现在p在左边所以需要转置
    R=np.linalg.inv(R)
    return  np.dot(points_3d, R) + T
def create_cube(location,rotation,scale,colletion):
    bpy.ops.mesh.primitive_cube_add(size=1, 
                                    location=location, 
                                    rotation=rotation,
                                    scale=scale)
    # 获取创建的立方体对象
    cube = bpy.context.active_object
    for col in cube.users_collection:
        col.objects.unlink(cube)  # 移除默认集合
    colletion.objects.link(cube)
def apply_boolean(obj,colletion):
    bool_modifier = obj.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_modifier.operation = 'DIFFERENCE'  # 或 'UNION' 或 'INTERSECT'        
    bool_modifier.operand_type='COLLECTION'
    bool_modifier.collection=colletion
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)    
    bpy.ops.object.modifier_apply(modifier="Boolean")
    # colletion.hide_viewport = True
def generate_cube_by_currentClasses(current_classes,facade_parameterization,poly_index,R,T,image_width,image_height,actual_width,actual_height,euler):
    Redundancy=0.2
    T_copy=T.copy()
    for c in current_classes:
        c_xywh=facade_parameterization[poly_index][c].copy()
        c_xywh[:,1]=image_height-c_xywh[:,3]-c_xywh[:,1]
        c_xywh=c_xywh-np.array([image_width/2,image_height/2,0,0])
        c_xywh=c_xywh/np.array([image_width,image_height,image_width,image_height])
        c_xywh=c_xywh*np.array([actual_width,actual_height,actual_width,actual_height])
        c_depth=facade_parameterization[poly_index][f"{c}Depth"]
        z_axis=R.T[2]
        if c=='glass':
            T_copy=T_copy-(facade_parameterization[poly_index]['windowDepth']*z_axis)
        
        c_locations = get_locations(c_xywh, R,T_copy)
        c_locations = c_locations+(-z_axis)*c_depth/2+Redundancy*(z_axis)/2
        collection_name=f"{c}Collection"
        if collection_name not in bpy.data.collections:
            c_collection = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(c_collection)
        else:
            c_collection = bpy.data.collections[collection_name]

        for i ,location in enumerate(c_locations):
            w,h=c_xywh[i][2:]
            create_cube(location=tuple(location),
                        rotation=euler,
                        scale=(w,h,c_depth+Redundancy),
                        colletion=c_collection)
def create_boolean_array(image_width, image_height, arrays):
    # 创建布尔数组，初始时全部为 False
    boolean_array = np.zeros((image_height, image_width), dtype=bool)
    for arr in arrays:
        ax, ay, aw, ah = arr
        boolean_array[ay:ay+ah, ax:ax+aw] = True
    return boolean_array
def ajust_UV(bm,bm_uv_layer,face,polygonPlane):
        for loop in face.loops:
            uv = loop[bm_uv_layer].uv
            vertex = loop.vert
            u, v = polygonPlane.point_to_UV(np.array(vertex.co))
            uv.x = u
            uv.y = v
        
def main():
    xyxy_window= np.array([[24, 76, 94, 169],
                           [172, 76, 244, 169],
                           [330, 76, 401, 169],
                           [24, 246, 94, 370],
                           [172, 246, 244, 370],
                           [330, 246, 401, 370],
                           [18,447, 90, 573],
                           [328,447, 400, 573],
                ])
    xyxy_glass=np.array([
                         [31, 86, 48, 108],
                         [52, 86, 68, 108],
                         [72, 86, 88, 108],
                         [31, 114, 48, 136],
                         [52, 114, 68, 136],
                         [72, 114, 88, 136],
                         [31, 139, 48, 162],
                         [52, 139, 68, 162],
                         [72, 139, 88, 162],
                         
                         [180, 86, 197, 108],
                         [201, 86, 217, 108],
                         [221, 86, 237, 108],
                         [180, 114, 197, 136],
                         [201, 114, 217, 136],
                         [221, 114, 237, 136],
                         [180, 139, 197, 162],
                         [201, 139, 217, 162],
                         [221, 139, 237, 162],

                         [336, 86, 353, 108],
                         [357, 86, 373, 108],
                         [377, 86, 393, 108],
                         [336, 114, 353, 136],
                         [357, 114, 373, 136],
                         [377, 114, 393, 136],
                         [336, 139, 353, 162],
                         [357, 139, 373, 162],
                         [377, 139, 393, 162],

                         [29, 256, 45, 279],
                         [48, 256, 65, 279],
                         [70, 256, 85, 279],
                         [29, 283, 45, 306],
                         [48, 283, 65, 306],
                         [70, 283, 85, 306],
                         [29, 311, 45, 333],
                         [48, 311, 65, 333],
                         [70, 311, 85, 333],
                         [29, 337, 45, 360],
                         [48, 337, 65, 360],
                         [70, 337, 85, 360],

                         [179, 256, 195, 279],
                         [198, 256, 215, 279],
                         [220, 256, 235, 279],
                         [179, 283, 195, 306],
                         [198, 283, 215, 306],
                         [220, 283, 235, 306],
                         [179, 311, 195, 333],
                         [198, 311, 215, 333],
                         [220, 311, 235, 333],
                         [179, 337, 195, 360],
                         [198, 337, 215, 360],
                         [220, 337, 235, 360],

                         [334, 256, 350, 279],
                         [353, 256, 370, 279],
                         [375, 256, 390, 279],
                         [334, 283, 350, 306],
                         [353, 283, 370, 306],
                         [375, 283, 390, 306],
                         [334, 311, 350, 333],
                         [353, 311, 370, 333],
                         [375, 311, 390, 333],
                         [334, 337, 350, 360],
                         [353, 337, 370, 360],
                         [375, 337, 390, 360],
                        
                         [334, 457, 350, 480],
                         [353, 457, 370, 480],
                         [375, 457, 390, 480],
                         [334, 484, 350, 507],
                         [353, 484, 370, 507],
                         [375, 484, 390, 507],
                         [334, 512, 350, 534],
                         [353, 512, 370, 534],
                         [375, 512, 390, 534],
                         [334, 538, 350, 561],
                         [353, 538, 370, 561],
                         [375, 538, 390, 561],

                         [29, 457, 45, 480],
                         [48, 457, 65, 480],
                         [70, 457, 85, 480],
                         [29, 484, 45, 507],
                         [48, 484, 65, 507],
                         [70, 484, 85, 507],
                         [29, 512, 45, 534],
                         [48, 512, 65, 534],
                         [70, 512, 85, 534],
                         [29, 538, 45, 561],
                         [48, 538, 65, 561],
                         [70, 538, 85, 561],
                ])
    xyxy_door=np.array([
        [178, 502, 235, 626],
                        ])
    xywh_window= xyxy_to_xywh(xyxy_window)
    xywh_glass= xyxy_to_xywh(xyxy_glass)
    xywh_door= xyxy_to_xywh(xyxy_door)
    image_path = r"E:\Desktop\facade\00472.jpg"
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
            'door':xywh_door,
            'windowDepth':0.1,
            'glassDepth':0.2,
            'doorDepth':0.7,
            },
        "3":{
            'window':xywh_window,
            'glass':xywh_glass,
            'door':xywh_door,
            'windowDepth':0.1,
            'glassDepth':0.2,
            'doorDepth':0.7,
            },
        "4":{
            'window':xywh_window,
            'glass':xywh_glass,
            'door':xywh_door,
            'windowDepth':0.1,
            'glassDepth':0.2,
            'doorDepth':0.7,

            },
        "5":{
            'window':xywh_window,
            'glass':xywh_glass,
            'door':xywh_door,
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
        # quaternion = basis_to_quaternion(R_before,R_after)      
        # cube.rotation_mode = 'QUATERNION'
        # cube.rotation_quaternion = mathutils.Quaternion(quaternion)
        # euler=quaternion.to_euler('XYZ')
        vertices = np.array([np.array(obj.data.vertices[idx].co) for idx in poly.vertices])
        actual_width = max(np.dot(vertices, x_axis)) - min(np.dot(vertices, x_axis))
        actual_height = max(np.dot(vertices, y_axis)) - min(np.dot(vertices, y_axis))
        

        polygonPlane = PolygonPlane(normal, center)
        if 'window' in current_classes:
            polygonPlane.window_depth=facade_parameterization[poly_index]['windowDepth']
        if 'glass' in current_classes:
            polygonPlane.glass_depth=facade_parameterization[poly_index]['glassDepth']
        if 'door' in current_classes:
            polygonPlane.door_depth=facade_parameterization[poly_index]['doorDepth']
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

    if 'window' in count_classes:
        window_collection = bpy.data.collections["windowCollection"]
        apply_boolean(obj,window_collection)
    if 'glass' in count_classes:
        glass_collection = bpy.data.collections["glassCollection"]
        apply_boolean(obj,glass_collection)  
    if 'door' in count_classes:
        door_collection = bpy.data.collections["doorCollection"]
        apply_boolean(obj,door_collection)

        
    
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
            if all(is_point_on_plane(point, polygonPlane.window_plane_equation) for point in vertices):
                window_poly_idxs.append(poly.index)           
            if all(is_point_on_plane(point, polygonPlane.glass_plane_equation) for point in vertices):
                glass_poly_idxs.append(poly.index)
            if all(is_point_on_plane(point, polygonPlane.door_plane_equation) for point in vertices):
                door_poly_idxs.append(poly.index)

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

    # bpy.ops.object.mode_set(mode='EDIT')
    """
    调整UV
    """
    # bpy.ops.object.mode_set(mode='EDIT')

    # bpy.ops.mesh.select_all(action='DESELECT')
    # bpy.ops.mesh.select_mode(type="FACE")
    # bpy.ops.object.mode_set(mode='OBJECT')
    # # uv_layer = obj.data.uv_layers.new(name="Custom_UV_Layer")
    # # obj.data.uv_layers.active = uv_layer
    # for face_index in window_poly_idxs:
    #     obj.data.polygons[face_index].select = True

    # for face_index in glass_poly_idxs:
    #     obj.data.polygons[face_index].select = True
    # bpy.ops.object.mode_set(mode='EDIT')
    # bpy.ops.uv.unwrap(method='CONFORMAL')
    # bpy.ops.uv.smart_project()
    # uv_layer_name = "UVMap"  # 替换为你的UV层名字
    # uv_layer = obj.data.uv_layers[uv_layer_name]
    # bpy.ops.object.mode_set(mode='EDIT')
    # bpy.ops.mesh.select_all(action='DESELECT')
    # bm = bmesh.from_edit_mesh(obj.data)
    # bm_uv_layer = bm.loops.layers.uv["UVMap"]
    # for polygonPlane in polygonPlaneList:
    #     for face_index in window_poly_idxs:
    #         face = bm.faces[face_index]
    #         vertices = np.array([np.array(loop.vert.co) for loop in face.loops])
    #         if all(is_point_on_plane(vertex, polygonPlane.window_plane_equation) for vertex in vertices):
    #             ajust_UV(bm,bm_uv_layer,face,polygonPlane)
                
    #     for face_index in glass_poly_idxs:
    #         face = bm.faces[face_index]
    #         vertices = np.array([np.array(loop.vert.co) for loop in face.loops])
    #         if all(is_point_on_plane(vertex, polygonPlane.glass_plane_equation) for vertex in vertices):
    #             ajust_UV(bm,bm_uv_layer,face,polygonPlane)
                       

    # bpy.ops.object.mode_set(mode='OBJECT')

    # material=bpy.data.materials.new("glass")  
    # material.use_nodes = True
    # # 获取材质的节点树
    # nodes = material.node_tree.nodes
    # # 创建图像纹理节点
    # image_texture = nodes.new(type='ShaderNodeTexImage')
    # # 加载图像
    # image_texture.image = image
    # # 获取纹理坐标节点
    # texture_coord = nodes.new(type='ShaderNodeTexCoord')
    # # 获取Principled BSDF节点
    # principled_bsdf = nodes.get("原理化 BSDF")
    # # 连接纹理坐标到图像纹理节点
    # material.node_tree.links.new(texture_coord.outputs['UV'], image_texture.inputs['Vector'])
    # # 连接图像纹理到 Principled BSDF 的 Base Color
    # material.node_tree.links.new(image_texture.outputs['Color'], principled_bsdf.inputs['Base Color'])    
    # principled_bsdf.inputs['Metallic'].default_value = 0.5 # 设置金属度，0.0为非金属，1.0为金属 
    # principled_bsdf.inputs['Roughness'].default_value = 0.0  # 设置粗糙度，0.0为光滑，1.0为粗糙
    # # 将材质应用到选中的物体
    # obj.data.materials.append(material)   
    # for face_index in glass_poly_idxs:
    #     add_material_by_faceIdx(obj,face_index,"glass")     
main()