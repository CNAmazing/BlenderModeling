import cv2
import numpy as np
def areaA_to_areaB_AffineTransform_By_points(pts_a,pts_b,a,b):
    M = cv2.getAffineTransform(pts_a, pts_b)

    # 获取图像 b 的尺寸
    h_b, w_b = b.shape[:2]

    # 应用仿射变换，将图像 a 中的三角形区域变换到图像 b 的三角形区域
    transformed_triangle = cv2.warpAffine(a, M, (w_b, h_b))

    # 创建一个掩码，用于提取变换后的三角形区域
    mask = np.zeros((h_b, w_b), dtype=np.uint8)  # 单通道掩码
    cv2.fillConvexPoly(mask, np.int32(pts_b), 255)  # 在图像 b 的三角形区域创建掩码

    # 将掩码扩展为 3 通道，以便与 transformed_triangle 进行按位与操作
    mask_3ch = cv2.merge([mask, mask, mask])

    # 将变换后的三角形区域融合到图像 b 中
    result = cv2.bitwise_and(transformed_triangle, mask_3ch)  # 提取变换后的三角形区域
    b = cv2.add(b, result)  # 将结果添加到图像 b 中
    return b
# 假设图像 a 和图像 b 已经加载
a = cv2.imread("poly6.jpg")
b = cv2.imread("poly_Texture28.jpg")

# 假设三角形区域的坐标已知
pts_a = np.float32([[0, 0.5], [10, 100], [100, 10]])  # 图像 a 中的三角形区域
pts_b = np.float32([[400, 400], [200.2, 0], [0, 200]])  # 图像 b 中的三角形区域

# 
b = areaA_to_areaB_AffineTransform_By_points(pts_a,pts_b,a,b)

# 保存或显示结果
cv2.imwrite("result.jpg", b)
cv2.imshow("Result", b)
cv2.waitKey(0)
cv2.destroyAllWindows()