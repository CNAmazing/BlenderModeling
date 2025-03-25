import math
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


