import math
import numpy as np
import bpy
import json
import os

OBJ_NAME='Cube'
FOLDER_PATH=os.path.join('output',OBJ_NAME)

def xyxy_to_xywh(bboxes):
    array=[]
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        array.append([x1, y1, x2 - x1, y2 - y1])
    return np.array(array)
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
def create_boolean_array( image_height, image_width,arrays):
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
def read_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误：{e}")
        return None  
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
        self.classes=None
        self.poly_idx=None
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
        将三维点转换为UV坐标
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
def compute_centroid(points):
    """
    计算三维点集的质心坐标
    :param points: 输入点集，形状为 (N, 3) 的 NumPy 数组或列表
    :return: 质心坐标 [Cx, Cy, Cz]
    """
    points = np.array(points)  # 确保转换为 NumPy 数组
    centroid = np.mean(points, axis=0)  # 沿第0轴（行方向）求均值
    return centroid  # 返回列表（可选）
def generate_cube_by_currentClasses(current_classes,facade_parameterization,poly_index,R,T,image_width,image_height,actual_width,actual_height,euler):
    Redundancy=0.2
    for c in current_classes:
        c_xywh=facade_parameterization[poly_index][c].copy()
        c_xywh[:,1]=image_height-c_xywh[:,3]-c_xywh[:,1]
        c_xywh=c_xywh-np.array([image_width/2,image_height/2,0,0])
        c_xywh=c_xywh/np.array([image_width,image_height,image_width,image_height])
        c_xywh=c_xywh*np.array([actual_width,actual_height,actual_width,actual_height])
        c_depth=facade_parameterization[poly_index][f"{c}Depth"]
        z_axis=R.T[2]
        if c=='glass':
            T=T-(facade_parameterization[poly_index]['windowDepth']*z_axis)
        else:
            T=T
        c_locations = get_locations(c_xywh, R,T)
        c_locations = c_locations+(-z_axis)*c_depth/2+Redundancy*z_axis/2
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
def generate_cube_by_currentClasses_PolyInfo(current_classes,polyInfo,euler):
    Redundancy=0.2
    image_width,image_height=polyInfo['poly_ImageSize']
    actual_width,actual_height=polyInfo['poly_ActualSize']
    R=np.array(polyInfo['basis']).T
    T=np.array(polyInfo['center'])
    for c in current_classes:
        c_xywh=np.array(polyInfo[c])
        c_xywh[:,1]=image_height-c_xywh[:,3]-c_xywh[:,1]
        c_xywh=c_xywh-np.array([image_width/2,image_height/2,0,0])
        c_xywh=c_xywh/np.array([image_width,image_height,image_width,image_height])
        c_xywh=c_xywh*np.array([actual_width,actual_height,actual_width,actual_height])
        c_depth=polyInfo[f"{c}Depth"]
        z_axis=R.T[2]
        if c=='glass':
            T=T-(polyInfo['windowDepth']*z_axis)
        else:
            T=T
        c_locations = get_locations(c_xywh, R,T)
        c_locations = c_locations+(-z_axis)*c_depth/2+Redundancy*z_axis/2
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
def get_basename_without_suffix(filepath):
    path = filepath
    filename = os.path.basename(path)  # 获取文件名 '00472.jpg'
    basename = filename.split('.')[0]
    return basename  # 返回文件名（不带扩展名）

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
def face_select_by_normal(v1,v2=(0,0,1),tolerance_degrees=1e-1):
    """
    判断两个向量是否正交。
    :param v1: 第一个向量，例如 (x1, y1, z1)
    :param v2: 第二个向量，例如 (x2, y2, z2)
    :param tolerance_degrees: 角度容差（默认1度）
    :return: 是否正交（True/False）
    """
    # 计算点积
    dot_product =abs(v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2])

    if dot_product<tolerance_degrees:
        return True
    else:   
        return False
def apply_boolean(obj,colletion):
    bool_modifier = obj.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_modifier.operation = 'DIFFERENCE'  # 或 'UNION' 或 'INTERSECT'        
    bool_modifier.operand_type='COLLECTION'
    bool_modifier.collection=colletion
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)    
    bpy.ops.object.modifier_apply(modifier="Boolean")
    # colletion.hide_viewport = True
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
def get_locations(xywh,R,T):
    #将矩形的左下角坐标转换为中心坐标
    
    center_coordinates = xywh[:, :2] + xywh[:, 2:] / 2
    points_3d = np.column_stack((center_coordinates, np.zeros(center_coordinates.shape[0])))
    # points_3d[:, [0, 1]] = points_3d[:, [1, 0]]
    #正常情况R*p  p为单个点 现在p在左边所以需要转置
    R=np.linalg.inv(R)
    return  np.dot(points_3d, R) + T
def add_material_by_faceIdx(obj ,faceIdx,material_name):
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

def is_normals_similar(v1, v2, tolerance_degrees=1):
    """
    判断两个向量是否相似。
    :param v1: 第一个向量，例如 (x1, y1, z1)
    :param v2: 第二个向量，例如 (x2, y2, z2)
    :param tolerance_degrees: 角度容差（默认1度）
    :return: 是否相似（True/False）
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
    
    # 判断是否相似
    return angle_deg < tolerance_degrees

