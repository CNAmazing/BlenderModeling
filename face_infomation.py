import bpy
import numpy as np
import mathutils
import builtins
class PolygonPlane():
    def __init__(self,normal, point):
        self.original_point = point
        self.normal = normal
        self.window_depth=None
        self.glass_depth=None
        self.door_depth=None
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
def rotation_matrix_to_quaternion(R):
    """将旋转矩阵转换为四元数"""
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return mathutils.Quaternion([qx, qy, qz, qw])
def basis_to_quaternion(R1,R2):
    """从两组基向量计算四元数"""
    # 构造变换前后的基向量矩阵
    E = R1.T
    F = R2.T

    # 计算旋转矩阵 R
    R = np.dot(F, np.linalg.inv(E))

    # 将旋转矩阵转换为四元数
    quaternion = rotation_matrix_to_quaternion(R)
    return quaternion
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
    bpy.ops.object.mode_set(mode='EDIT')
    # 获取网格数据
    mesh = obj.data
    # 切换到对象模式以访问面片数据
    bpy.ops.object.mode_set(mode='OBJECT')
    # 获取材质索引
    material_index = obj.data.materials.find(material_name)
    # 获取指定面片（例如索引为0的面片）
    mesh.polygons[faceIdx].material_index = material_index
    # 切换回物体模式
    bpy.ops.object.mode_set(mode='OBJECT')
 # 临时禁用 print
def get_locations(xywh,R,T,image_width,image_height):
    #将矩形的左上角坐标转换为中心坐标
    xywh=xywh-np.array([image_width/2,image_height/2,0,0])
    center_coordinates = xywh[:, :2] + xywh[:, 2:] / 2
    points_3d = np.column_stack((center_coordinates, np.zeros(center_coordinates.shape[0])))
    points_3d[:, [0, 1]] = points_3d[:, [1, 0]]
    #正常情况R*p  p为单个点 现在p在左边所以需要转置
    return  np.dot(points_3d, R.T) + T
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
    colletion.hide_viewport = True
"""
禁用 print
"""
# original_print = builtins.print
# builtins.print = lambda *args, **kwargs: None

facade_parameterization={
    "1":{'window':[[1, 1, 3, 3],[5, 1, 3, 3],[1, 5, 3, 3],[5, 5, 3, 3]],
         'door':[[4, 9, 3, 1]],
         'glass':[[1.25, 1.25, 1, 1],[2.75, 1.25, 1, 1],[1.25, 2.75, 1, 1],[2.75, 2.75, 1, 1]],
         'windowDepth':0.25,
         'glassDepth':0.24,
         'doorDepth':0.6,
         'xy_axis':[[0,0,-1],[1,0,0]]},
    "2":{'window':[[1, 1, 3, 3],[5, 1, 3, 3],[1, 5, 3, 3],[5, 5, 3, 3]],
         'door':[[4, 9, 3, 1]],
         'glass':[[1.25, 1.25, 1, 1],[2.75, 1.25, 1, 1],[1.25, 2.75, 1, 1],[2.75, 2.75, 1, 1]],
         'windowDepth':0.25,
         'glassDepth':0.24,
         'doorDepth':0.6,
         'xy_axis':[[0,0,-1],[0,-1,0]]},
    "4":{'window':[[1, 1, 3, 3],[5, 1, 3, 3],[1, 5, 3, 3],[5, 5, 3, 3]],
         'door':[[4, 9, 3, 1]],
         'glass':[[1.25, 1.25, 1, 1],[2.75, 1.25, 1, 1],[1.25, 2.75, 1, 1],[2.75, 2.75, 1, 1]],
         'windowDepth':0.25,
         'glassDepth':0.24,
         'doorDepth':0.6,
         'xy_axis':[[0,0,-1],[0,1,0]]},
    "5":{'window':[[1, 1, 3, 3],[5, 1, 3, 3],[1, 5, 3, 3],[5, 5, 3, 3]],
         'door':[[4, 9, 3, 1]],
         'glass':[[1.25, 1.25, 1, 1],[2.75, 1.25, 1, 1],[1.25, 2.75, 1, 1],[2.75, 2.75, 1, 1]],
         'windowDepth':0.25,
         'glassDepth':0.24,
         'doorDepth':0.6,
         'xy_axis':[[0,0,-1],[-1,0,0]]},
}

image_width = 10
image_height = 10
# rectangles = rectangles / np.array([image_width, image_height, image_width, image_height])
# actually_width = 10
# actually_height = 10
# rectangles = rectangles * np.array([actually_width, actually_height, actually_width, actually_height])


# 1. 创建一个新的集合
windows_collection = bpy.data.collections.new("windowsCollection")
doors_collection = bpy.data.collections.new("doorsCollection")
glasses_collection = bpy.data.collections.new("glassesCollection")
# 2. 将集合添加到当前场景
bpy.context.scene.collection.children.link(windows_collection)
bpy.context.scene.collection.children.link(doors_collection)
bpy.context.scene.collection.children.link(glasses_collection)

obj = bpy.context.scene.objects.get("Cube")  

if obj.type != 'MESH':
    raise ValueError(f"{obj.name} 不是网格对象")
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
polygonPlaneList=[]
total_classes=['window','glass','door']
# 遍历所有面片
for poly in obj.data.polygons:
    # print(f"面片索引: {poly.index}")
    # print(f"边界索引: {poly.loop_indices}") 
    # print(f"边索引键值对: {poly.edge_keys}")
    # 获取面片的法线
    if str(poly.index) not in facade_parameterization:
        continue

    current_classes=[key for key in total_classes if key in facade_parameterization[str(poly.index)]]
    normal = poly.normal
    center = poly.center
    normal =np.array(normal)
    center = np.array(center)
    
    x_axis,y_axis=facade_parameterization[str(poly.index)]['xy_axis']
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
    quaternion = basis_to_quaternion(R_before,R_after)      
    # cube.rotation_mode = 'QUATERNION'
    # cube.rotation_quaternion = mathutils.Quaternion(quaternion)
    euler=quaternion.to_euler('XYZ')

    polygonPlane = PolygonPlane(normal, center)
    polygonPlane.window_depth=facade_parameterization[str(poly.index)]['windowDepth']
    polygonPlane.glass_depth=facade_parameterization[str(poly.index)]['glassDepth']
    polygonPlane.door_depth=facade_parameterization[str(poly.index)]['doorDepth']
    polygonPlaneList.append(polygonPlane)
    
    windows_xywh=facade_parameterization[str(poly.index)]['window']
    glass_xywh=facade_parameterization[str(poly.index)]['glass']
    door_xywh=facade_parameterization[str(poly.index)]['door']

    

    window_depth=facade_parameterization[str(poly.index)]['windowDepth'] 
    glass_depth=facade_parameterization[str(poly.index)]['glassDepth']
    door_depth=facade_parameterization[str(poly.index)]['doorDepth']

    windows_locations = get_locations(windows_xywh,R,T,image_width,image_height)
    door_locations = get_locations(door_xywh,R,T,image_width,image_height)
    glass_locations = get_locations(glass_xywh,R,T-(window_depth*z_axis),image_width,image_height)

    windows_locations = windows_locations+(-z_axis)*window_depth/2
    door_locations = door_locations+(-z_axis)*door_depth/2
    glass_locations = glass_locations+(-z_axis)*glass_depth/2

    for i ,location in enumerate(windows_locations):
        w,h=windows_xywh[i][2:]
        create_cube(location=tuple(location),
                    rotation=euler,
                    scale=(h,w,window_depth),
                    colletion=windows_collection)
        
    for i,location in enumerate(glass_locations):
        w,h=glass_xywh[i][2:]
        create_cube(location=tuple(location),
                    rotation=euler,
                    scale=(h,w,glass_depth),
                    colletion=glasses_collection)

    for i ,location in enumerate(door_locations):
        w,h=door_xywh[i][2:]
        create_cube(location=tuple(location),
                    rotation=euler,
                    scale=(h,w,door_depth),
                    colletion=doors_collection)
    
                    

'''
应用布尔运算
'''
apply_boolean(obj,windows_collection)
apply_boolean(obj,doors_collection)
apply_boolean(obj,glasses_collection)

facade_poly_idxs=[]
windowSide_poly_idxs=[]   
window_poly_idxs=[]
glass_poly_idxs=[]
door_poly_idxs=[]

for polygonPlane in polygonPlaneList:
    # print(f"facade_plane_equation: {polygonPlane.facade_plane_equation}")
    # print(f"window_plane_equation: {polygonPlane.window_plane_equation}")
    # 获取物体的变换矩阵
    for poly in obj.data.polygons:
        # 获取面片索引
        # print(f"面片索引: {poly.index}")
        # 将局部空间坐标转换为世界空间坐标
        vertices = [obj.data.vertices[idx].co for idx in poly.vertices]
        print(f"vertices{vertices}")
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
             'windows':(255,0,0),
             'glass':(0,0,255),
             'door':(250,86,53),
             'balcony':(32,140,62),}

create_material(color_dict)
add_material_to_object(obj,color_dict)

# 获取指定面片（例如索引为0的面片）
# for face_index in facade_poly_idxs:
#     add_material_by_faceIdx(obj,face_index,"facade")
# for face_index in window_poly_idxs:
#     add_material_by_faceIdx(obj,face_index,"windows")
# for face_index in glass_poly_idxs:
#     add_material_by_faceIdx(obj,face_index,"glass")
# for face_index in door_poly_idxs:
#     add_material_by_faceIdx(obj,face_index,"door")
# for face_index in windowSide_poly_idxs:
#     add_material_by_faceIdx(obj,face_index,"glass")

# 进入编辑模式
bpy.ops.object.mode_set(mode='EDIT')
# 打开 UV 编辑器并展开 UV
# 选择所有面
# bpy.ops.mesh.select_all(action='SELECT')
uv_layer = obj.data.uv_layers.new(name="Custom_UV_Layer")
obj.data.uv_layers.active = uv_layer
bpy.ops.mesh.select_all(action='SELECT')
for face_index in window_poly_idxs:
    obj.data.polygons[face_index].select = True
for face_index in glass_poly_idxs:
    obj.data.polygons[face_index].select = True
# bpy.ops.uv.lightmap_pack()  # Lightmap Pack
# bpy.ops.uv.unwrap()
# bpy.ops.uv.unwrap(method='SMART')
# bpy.ops.uv.smart_project(angle_limit=10, island_margin=0.02)
# 返回对象模式

# bpy.ops.object.mode_set(mode='OBJECT')

# material=bpy.data.materials.get("glass")  
# material.use_nodes = True
# # 获取材质的节点树
# nodes = material.node_tree.nodes
# # 创建图像纹理节点
# image_texture = nodes.new(type='ShaderNodeTexImage')
# # 加载图像
# image_path = "E:/Desktop/glass.jpg"  # 替换为图片的路径
# image = bpy.data.images.load(image_path)
# image_texture.image = image
# # 获取纹理坐标节点
# texture_coord = nodes.new(type='ShaderNodeTexCoord')
# # 获取Principled BSDF节点
# principled_bsdf = nodes.get("原理化 BSDF")
# # 连接纹理坐标到图像纹理节点
# material.node_tree.links.new(texture_coord.outputs['UV'], image_texture.inputs['Vector'])
# # 连接图像纹理到 Principled BSDF 的 Base Color
# material.node_tree.links.new(image_texture.outputs['Color'], principled_bsdf.inputs['Base Color'])
# principled_bsdf.inputs['Base Color'].default_value = (0.0, 0.0, 1.0, 1.0)  # 设置基础颜色
# principled_bsdf.inputs['Metallic'].default_value = 5.0  # 设置金属度，0.0为非金属，1.0为金属
# principled_bsdf.inputs['Roughness'].default_value = 0.0  # 设置粗糙度，0.0为光滑，1.0为粗糙

# 将材质应用到选中的物体
# obj.data.materials.append(material)

# 获取 UV 层

uv_layer_data = obj.data.uv_layers.active.data
for face_index in glass_poly_idxs:
    face= obj.data.polygons[face_index]
    # uv_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for i,loop_index in enumerate(face.loop_indices):
        print(f"loop_index:{loop_index}")
        uv=uv_layer_data[loop_index].uv
        # uv.x, uv.y = uv_coords[i]
        print('uv.x:',uv.x)
        print('uv.y:',uv.y)
        print('uv:',uv)
