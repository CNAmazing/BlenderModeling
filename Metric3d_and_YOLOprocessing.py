import onnxruntime as ort
import numpy as np
import cv2
import os
from typing import Tuple, Dict, List
from matplotlib import pyplot as plt
import sys
cwd = os.getcwd()
sys.path.append(cwd)
from Hw_Tools import *

def prepare_input(
    rgb_image: np.ndarray, input_size: Tuple[int, int]
) -> Tuple[Dict[str, np.ndarray], List[int]]:

    h, w = rgb_image.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(
        rgb_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
    )

    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb: np.ndarray = cv2.copyMakeBorder(
        rgb,
        pad_h_half,
        pad_h - pad_h_half,
        pad_w_half,
        pad_w - pad_w_half,
        cv2.BORDER_CONSTANT,
        value=padding,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    onnx_input = {
        "pixel_values": np.ascontiguousarray(
            np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32
        ),  # 1, 3, H, W
    }
    return onnx_input, pad_info

def Metric3d_Processing(onnx_model=None, input_image=None):
    
        # 分解路径
    dir_path = os.path.dirname(input_image)  # 获取目录路径: "output\Cube.001\images"
    base_name = os.path.basename(input_image)  # 获取文件名: "poly0.jpg"
    name_without_ext = os.path.splitext(base_name)[0]  # 去除扩展名: "poly0"
    ## Dummy Test
    B = 1
    if "vit" in onnx_model:
        input_size = (616, 1064)  # [H, W]
        dummy_image = np.zeros([B, 3, input_size[0], input_size[1]], dtype=np.float32)
    else:
        input_size = (544, 1216)  # [H, W]
        dummy_image = np.zeros([B, 3, input_size[0], input_size[1]], dtype=np.float32)

    providers = [
        (
            "CUDAExecutionProvider",
            {"cudnn_conv_use_max_workspace": "0", "device_id": str(0)},
        )
    ]
    # providers = [("TensorrtExecutionProvider", {'trt_engine_cache_enable': True, 'trt_fp16_enable': True, 'device_id': 0, 'trt_dla_enable': False})]
    ort_session = ort.InferenceSession(onnx_model, providers=providers)
    outputs = ort_session.run(None, {"pixel_values": dummy_image})

    print(
        f"The actual output of onnxruntime session for the dummy set: outputs[0].shape={outputs[0].shape}"
    )

    ## Real Test
    rgb_image = cv2.imread(input_image)[:, :, ::-1]  # BGR to RGB
    original_shape = rgb_image.shape[:2]
    onnx_input, pad_info = prepare_input(rgb_image, input_size)
    outputs = ort_session.run(None, onnx_input)
    depth = outputs[0].squeeze()  # [H, W]
    normal=outputs[1].squeeze()  # [3,H, W]
    # Reshape the depth to the original size
    depth = depth[
        pad_info[0] : input_size[0] - pad_info[1],
        pad_info[2] : input_size[1] - pad_info[3],
    ]
    normal=normal[
        :,
        pad_info[0] : input_size[0] - pad_info[1],
        pad_info[2] : input_size[1] - pad_info[3],
    ]
    normal = np.transpose(normal, (1, 2, 0))  # [H,W,3]

    normal = cv2.resize(
        normal, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR
    )
    normal=(normal+1)/2
    normal = (normal * 255).astype(np.uint8)
    depth = cv2.resize(
        depth, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST
    )

    # plt.subplot(1, 3, 1)
    # plt.imshow(normal)
    
    # plt.subplot(1, 3, 2)
    # plt.imshow(depth)
    
    # plt.subplot(1, 3, 3)
    # plt.imshow(rgb_image)
    # plt.show()
    normal_output_Path= os.path.join(dir_path, f"{name_without_ext}_normal.jpg")
    depth_output_Path= os.path.join(dir_path, f"{name_without_ext}_depth.jpg")
    cv2.imwrite(normal_output_Path, normal)
    cv2.imwrite(depth_output_Path, depth)


 
 
class YOLO11:
    """YOLO11 目标检测模型类，用于处理推理和可视化。"""
    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        """
        初始化 YOLO11 类的实例。
        参数：
            onnx_model: ONNX 模型的路径。
            input_image: 输入图像的路径。
            confidence_thres: 用于过滤检测结果的置信度阈值。
            iou_thres: 非极大值抑制（NMS）的 IoU（交并比）阈值。
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
 
        # 加载类别名称
        self.classes = CLASS_NAMES
 
        # 为每个类别生成一个颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
 
    def preprocess(self):
        """
        对输入图像进行预处理，以便进行推理。
        返回：
            image_data: 经过预处理的图像数据，准备进行推理。
        """
        # 使用 OpenCV 读取输入图像
        self.img = cv2.imread(self.input_image)
        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]
 
        # 将图像颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
 
        # 保持宽高比，进行 letterbox 填充, 使用模型要求的输入尺寸
        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, new_shape=(self.input_width, self.input_height))
 
        # 通过除以 255.0 来归一化图像数据
        image_data = np.array(img) / 255.0
 
        # 将图像的通道维度移到第一维
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先
 
        # 扩展图像数据的维度，以匹配模型输入的形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
 
        # 返回预处理后的图像数据
        return image_data
 
 
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """
        将图像进行 letterbox 填充，保持纵横比不变，并缩放到指定尺寸。
        """
        shape = img.shape[:2]  # 当前图像的宽高
        print(f"Original image shape: {shape}")
 
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
 
        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 选择宽高中最小的缩放比
        if not scaleup:  # 仅缩小，不放大
            r = min(r, 1.0)
 
        # 缩放后的未填充尺寸
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
 
        # 计算需要的填充
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算填充的尺寸
        dw /= 2  # padding 均分
        dh /= 2
 
        # 缩放图像
        if shape[::-1] != new_unpad:  # 如果当前图像尺寸不等于 new_unpad，则缩放
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
 
        # 为图像添加边框以达到目标尺寸
        top, bottom = int(round(dh)), int(round(dh))
        left, right = int(round(dw)), int(round(dw))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        if img.shape[0] != new_shape[0] or img.shape[1] != new_shape[1]:
            img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
        print(f"Final letterboxed image shape: {img.shape}")
 
        return img, (r, r), (dw, dh)
    def postprocess(self, input_image, output):
        """
        对模型输出进行后处理，以提取边界框、分数和类别 ID。
        参数：
            input_image (numpy.ndarray): 输入图像。
            output (numpy.ndarray): 模型的输出。
        返回：
            numpy.ndarray: 包含检测结果的输入图像。
        """
        # 转置并压缩输出，以匹配预期形状
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []
        # 计算缩放比例和填充
        ratio = self.img_width / self.input_width, self.img_height / self.input_height
 
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
 
                # 将框调整到原始图像尺寸，考虑缩放和填充
                x -= self.dw  # 移除填充
                y -= self.dh
                x /= self.ratio[0]  # 缩放回原图
                y /= self.ratio[1]
                w /= self.ratio[0]
                h /= self.ratio[1]
                left = int(x - w / 2)
                top = int(y - h / 2)
                width = int(w)
                height = int(h)
 
                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        print('indices',indices)
        result_boxes = []
        result_scores = []
        result_class_ids = []
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            result_boxes.append(box)
            result_scores.append(score)
            result_class_ids.append(class_id)
            self.draw_detections(input_image, box, score, class_id)
        result={
            'boxes':result_boxes,
            'scores':result_scores,
            'class_ids':result_class_ids
            }
        return result, input_image
    def draw_detections(self, img, box, score, class_id):
        """
        在输入图像上绘制检测到的边界框和标签。
        参数：
            img: 用于绘制检测结果的输入图像。
            box: 检测到的边界框。
            score: 对应的检测分数。
            class_id: 检测到的目标类别 ID。
        
        返回：
            None
        """
        # 提取边界框的坐标
        x1, y1, w, h = box
 
        # 获取类别对应的颜色
        color = self.color_palette[class_id]
 
        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
 
        # 创建包含类别名和分数的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"
 
        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
 
        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
 
        # 绘制填充的矩形作为标签文本的背景
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
 
        # 在图像上绘制标签文本
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
 
 
    def main(self):
        # 使用 ONNX 模型创建推理会话，自动选择CPU或GPU
        session = ort.InferenceSession(
            self.onnx_model, 
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"],
        )
        # 打印模型的输入尺寸
        print("YOLO11 🚀 目标检测 ONNXRuntime")
        print("模型名称：", self.onnx_model)
        
        # 获取模型的输入形状
        model_inputs = session.get_inputs()
        input_shape = model_inputs[0].shape  
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        print(f"模型输入尺寸：宽度 = {self.input_width}, 高度 = {self.input_height}")
 
        # 预处理图像数据，确保使用模型要求的尺寸 (640x640)
        img_data = self.preprocess()
       
        # 使用预处理后的图像数据运行推理
        outputs = session.run(None, {model_inputs[0].name: img_data})
 
        # 对输出进行后处理以获取输出图像
        return self.postprocess(self.img, outputs)  # 输出图像
class YOLO11_glass(YOLO11):
    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        super().__init__(onnx_model, input_image, confidence_thres, iou_thres)
        self.color_palette= {0: 'glass', }
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
    """YOLO11 目标检测模型类，用于处理推理和可视化。"""
    def preprocess(self):
        """
        对输入图像进行预处理，以便进行推理。
        返回：
            image_data: 经过预处理的图像数据，准备进行推理。
        """
        self.img = self.input_image
        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.input_image.shape[:2]
 
        # 将图像颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
 
        # 保持宽高比，进行 letterbox 填充, 使用模型要求的输入尺寸
        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, new_shape=(self.input_width, self.input_height))
 
        # 通过除以 255.0 来归一化图像数据
        image_data = np.array(img) / 255.0
 
        # 将图像的通道维度移到第一维
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先
 
        # 扩展图像数据的维度，以匹配模型输入的形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
 
        # 返回预处理后的图像数据
        return image_data
 
    def letterbox(self, img, new_shape=(256, 256), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """
        将图像进行 letterbox 填充，保持纵横比不变，并缩放到指定尺寸。
        """
        shape = img.shape[:2]  # 当前图像的宽高
        print(f"Original image shape: {shape}")
 
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
 
        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 选择宽高中最小的缩放比
        if not scaleup:  # 仅缩小，不放大
            r = min(r, 1.0)
 
        # 缩放后的未填充尺寸
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
 
        # 计算需要的填充
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算填充的尺寸
        dw /= 2  # padding 均分
        dh /= 2
 
        # 缩放图像
        if shape[::-1] != new_unpad:  # 如果当前图像尺寸不等于 new_unpad，则缩放
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
 
        # 为图像添加边框以达到目标尺寸
        top, bottom = int(round(dh)), int(round(dh))
        left, right = int(round(dw)), int(round(dw))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        if img.shape[0] != new_shape[0] or img.shape[1] != new_shape[1]:
            img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
        print(f"Final letterboxed image shape: {img.shape}")
 
        return img, (r, r), (dw, dh)
    
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
        images_dir = os.path.join(folder_name, "images")
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
CLASS_NAMES = {
    0: 'window',   # 类别 0 名称
    1: 'balcony',   # 类别 1 名称
    2: 'door'    # 类别 1 名称
                        # 可以添加更多类别...
}
TOTAL_CLASSES=['window','door','glass']
def main():
    input_folder = "output\Cube.001"
    jps_paths,basenames=get_jpg_paths(input_folder)
    json_path=os.path.join(input_folder,'data.json')
    poly_dict=read_json(json_path)

    #删除重复生成的box
    for poly in poly_dict.values():
        for cls in TOTAL_CLASSES:
            poly.pop(cls, None)
    #读取每个图像进行Metric3d和YOLO处理   
    for image_path,basename in zip(jps_paths, basenames):
        
        # 处理每个图片
        Metric3d_Processing(onnx_model="checkpoint\Metric3d_vit_small.onnx",
                            input_image=image_path
            )
        
        facade_detection = YOLO11( onnx_model="checkpoint\YOLO_window.onnx",
                            input_image=image_path,
                            confidence_thres=0.5,
                            iou_thres=0.25,
            )
        facade_Params={
            'window':[],
            'glass':[],
            'door':[],
        }

        facade_result,output_image = facade_detection.main()
        image = cv2.imread(image_path)
        # 获取输入图像的高度和宽度

        for i in range(len(facade_result['boxes'])):
            box = facade_result['boxes'][i]
            score = facade_result['scores'][i]
            class_id = facade_result['class_ids'][i]
            facade_Params[CLASS_NAMES[class_id]].append(box)

            if CLASS_NAMES[class_id] == 'window':
                window_x1, window_y1, window_w, window_h = box
                glass_detection = YOLO11_glass( onnx_model="checkpoint\YOLO_glass.onnx",
                            input_image=image[window_y1:window_y1+window_h,window_x1:window_x1+window_w,: ],
                            confidence_thres=0.5,
                            iou_thres=0.25,
                            )
                glass_result,glass_output_image = glass_detection.main()
                for j in range(len(glass_result['boxes'])):
                    g_box= glass_result['boxes'][j]
                    g_box[0] += window_x1
                    g_box[1] += window_y1
                    facade_Params['glass'].append(g_box)
        
        #给poly_dict添加数据
        for key, value in facade_Params.items():
            if value:
                poly_dict[basename][key]=value
        
    """
    保存为json文件
    """
    with open(json_path, "w") as json_file:
        json.dump(poly_dict, json_file, indent=4)
    
    # """
    # Metric3D Processing只处理单图像
    # """
    # Metric3d_Processing(
    # onnx_model="checkpoint\Metric3d_large.onnx",
    # input_image="output\poly0_.jpg"
    # )

    # """
    # YOLO Processing
    # 只处理单图像
    # """
    # detection = YOLO11(
    #     onnx_model="checkpoint\YOLO_window.onnx",
    #     input_image="output\poly0_.jpg",
    #     confidence_thres=0.25,
    #     iou_thres=0.45,
    # )
    # """"
    # result-————————————>字典
    # 'boxes': 检测到的边界框列表
    # 'scores': 对应的检测分数列表
    # 'class_ids': 检测到的目标类别 ID 列表
    # """
    # result,output_image = detection.main()
    
main()
print("Done!")
