import time
import cv2
import numpy
import numpy as np
from typing import Tuple
import onnxruntime as rt
from bytetrack_init import bytetrack,make_parser
from yolov8onnx.utils import xywh2xyxy, nms
from ultralytics.trackers import BYTETracker
from test import SR

cgr_model=rt.InferenceSession("model/rtdetr-best.onnx")
input_name = cgr_model.get_inputs()[0].name
label_name = cgr_model.get_outputs()[0].name
pose_model=rt.InferenceSession("model/yolov8n-pose.onnx")
input_names = pose_model.get_inputs()[0].name
label_names = pose_model.get_outputs()[0].name
args = make_parser().parse_args()
tracker = BYTETracker(args, frame_rate=30)
tracker_cgr=BYTETracker(args, frame_rate=60)


def smooth_detections(detections, kf):

    smoothed_detections = []
    for detection in detections:
        prediction = kf.predict()
        if prediction is not None:
            smoothed_detections.append(prediction)
        kf.update(detection)
    return smoothed_detections

def prepare_input(image):
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize input image
    input_img,border = proportional_resize_with_padding(input_img, (640,640))

    # Scale input pixel values to 0 to 1
    input_img = input_img / 255.0
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
    return input_tensor,border


def process_output(output,conf_threshold,iou_threshold,img,inputimg,border):
    predictions = np.squeeze(output[0]).T

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        return [], [], []

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes =extract_boxes(predictions,img,inputimg,border)
    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indices = nms(boxes, scores, iou_threshold)
    # indices = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold)

    return boxes[indices], scores[indices], class_ids[indices]


def extract_boxes(predictions,ori,inputimg,border):
    # Extract boxes from predictions
    boxes = predictions[:, :4]
    boxes = bbox_cxcywh_to_xyxy(boxes)
    # Scale boxes to original image dimensions
    boxes = rscale_box_with_padding((border[4],border[5]),boxes,(ori.shape[1],ori.shape[0]),border)
    # boxes = resize_box_with_padding(boxes,ori,inputimg)
    # Convert boxes to xyxy format
    return boxes


def preprocess(image_path, ort_model):
    # 加载图片
    image = image_path

    # 获取图像宽高
    image_h, image_w = image.shape[:2]

    # 获取所有输入节点信息
    input_nodes = ort_model.get_inputs()

    # 筛选出名称为"images"的输入节点信息
    input_node = next(filter(lambda n: n.name == "images", input_nodes), None)

    if input_node is None:
        raise ValueError("No input node with name 'images'")

    # 输入尺寸
    input_shape = input_node.shape[-2:]
    input_h, input_w = 640,640

    # 缩放因子
    ratio_h = input_h / image_h
    ratio_w = input_w / image_w

    # 预处理步骤
    img = cv2.resize(image, (0, 0), fx=ratio_w, fy=ratio_h, interpolation=2)
    img = img[:, :, ::-1] / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = np.ascontiguousarray(img, dtype=np.float32)

    return image, img, (image_h, image_w)
def bbox_cxcywh_to_xyxy(boxes):

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return np.stack([x1, y1, x2, y2], axis=1)
def postprocess(outs, conf_thres, im_shape):
    boxes, scores = outs[:, :4], outs[:, 4:]

        # 根据 scores 数值分布判断是否进行归一化处理
    if not (np.all((scores > 0) & (scores < 1))):
        print("normalized value >>>")
        scores = 1 / (1 + np.exp(-scores))

    boxes = bbox_cxcywh_to_xyxy(boxes)
    _max = scores.max(-1)
    _mask = _max > conf_thres
    boxes, scores = boxes[_mask], scores[_mask]
    labels, scores = scores.argmax(-1), scores.max(-1)

    # 对边框信息进行尺度归一化
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = np.floor(np.minimum(np.maximum(1, x1 * im_shape[1]), im_shape[1] - 1)).astype(int)
    y1 = np.floor(np.minimum(np.maximum(1, y1 * im_shape[0]), im_shape[0] - 1)).astype(int)
    x2 = np.ceil(np.minimum(np.maximum(1, x2 * im_shape[1]), im_shape[1] - 1)).astype(int)
    y2 = np.ceil(np.minimum(np.maximum(1, y2 * im_shape[0]), im_shape[0] - 1)).astype(int)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    return boxes,scores
def proportional_resize_with_padding(img,new_shape: Tuple[int, int])-> np.array:
    try:
        # 获取原始图片的宽高
        original_height, original_width, _ = img.shape

        # 计算宽高比例
        width_ratio = new_shape[1] / original_width
        height_ratio = new_shape[0] / original_height

        # 选择缩放比例较小的那个，以保持原始图像的纵横比
        resize_ratio = min(width_ratio, height_ratio)

        # 计算缩放后的尺寸
        new_width = int(original_width * resize_ratio)
        new_height = int(original_height * resize_ratio)

        # 进行缩放
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 计算边框大小
        top = (new_shape[0] - new_height) // 2
        bottom = new_shape[0] - new_height - top
        left = (new_shape[1] - new_width) // 2
        right = new_shape[1] - new_width - left

        # 添加边框
        padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        return padded_img,(top,bottom,left,right,new_width,new_height)
    except Exception as e:
        print(f"Error: {e}")
        return None


def xywh2xyxy_rescale(x,scale,is_scale):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    if is_scale:
        y[..., 0] = x[..., 0]*scale
        y[..., 1] = x[..., 1]*scale
        y[..., 2] = (x[..., 0] + x[..., 2])*scale
        y[..., 3] = (x[..., 1] + x[..., 3])*scale
    else:
        y[..., 0] = x[..., 0]
        y[..., 1] = x[..., 1]
        y[..., 2] = (x[..., 0] + x[..., 2])
        y[..., 3] = (x[..., 1] + x[..., 3])
    return y

def rscale_box_with_padding(original_size,boxes, target_size,border):
    """
    Resize a detection box from original size to target size with padding.

    Parameters:
        original_size (tuple): A tuple (width, height) representing the original image size.
        original_box_xyxy (tuple): A tuple (x1, y1, x2, y2) representing the bounding box coordinates
                                   in the original image before padding.
        target_size (tuple): A tuple (width, height) representing the target image size after padding.

    Returns:
        tuple: A tuple (x1, y1, x2, y2) representing the bounding box coordinates in the target image.
    """
    original_width, original_height = original_size
    target_width, target_height = target_size


    # Calculate scaling factors in x and y directions
    scale_x = target_width / original_width
    scale_y = target_height / original_height

    # Adjust the box coordinates to account for padding
    boxes-= np.array([border[2],border[0],border[2],border[0]])
    boxes *= np.array([scale_x,scale_y,scale_x,scale_y])
    return boxes


def cgr_detect_with_onnx(frame):
    frame = SR(frame)
    [height, width, _] = frame.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame
    scale = length / 640
    image = cv2.resize(image, (640, 640))
    image = image.astype(np.float32) / 255.0
    input_img = np.transpose(image, [2, 0, 1])
    input_img = input_img[np.newaxis, :, :, :]
    # Create tensor from external memory
    outputs = cgr_model.run([label_name], {input_name: input_img})
    output_buffer=np.transpose(outputs[0],[0,2,1])
    boxes, scores= postprocess(outputs[0].squeeze(),0.35,(length/3,length/3))
    # boxes, result = bytetrack(boxes, scores,class_ids, tracker_cgr)
    if isinstance(boxes, numpy.ndarray) and boxes.shape[0]!=0:
        # boxes = xywh2xyxy_rescale(boxes, 0, False)
        return boxes,scores
    else:
        return [],[]

def pose_estimate_with_onnx(frame):
    [height, width, _] = frame.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame
    scale = length / 640
    image = cv2.resize(image, (640, 640))
    image = image.astype(np.float32) / 255.0
    input_img = np.transpose(image, [2, 0, 1])
    input_img = input_img[np.newaxis, :, :, :]
    # blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    # 基于OpenVINO实现推理计算
    outputs = pose_model.run([label_names],{input_names:input_img})
    outputs = np.transpose(outputs[0],[0,2,1])

    classes_scores = outputs[:, :, 4]
    key_points = outputs[:, :, 5:]

    # Create a mask to filter rows based on classes_scores >= 0.5
    mask = classes_scores >= 0.5

    # Use the mask to get the filtered_outputs array
    filtered_outputs = outputs[mask]

    # Calculate boxes using vectorized operations
    boxes = filtered_outputs[:, 0:4] - np.column_stack(
        [(0.5 * filtered_outputs[:, 2]), (0.5 * filtered_outputs[:, 3]), np.zeros_like(filtered_outputs[:, 2]),
         np.zeros_like(filtered_outputs[:, 3])])

    # Extract scores and key points
    scores = filtered_outputs[:, 4]
    preds_kpts = key_points[mask]

    # Optionally, convert the boxes list to a NumPy array

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.5, 0.5)

    box = np.array(boxes)[result_boxes].reshape(-1, 4)
    box = xywh2xyxy_rescale(box, scale,True)
    scores = np.array(scores)[result_boxes]
    cls=numpy.zeros(box.shape[0])
    box,result = bytetrack(box, scores,cls,tracker)
    if isinstance(box,numpy.ndarray) and box.shape[0]!=0:
        box= xywh2xyxy_rescale(box,scale,False)
    kpts = np.array(preds_kpts)[result_boxes].reshape(-1,17,3)*scale
    return box,scores,result,kpts

def cgr_update(box,score):
    cgr_cls=numpy.zeros(box.shape[0])
    box,result = bytetrack(box, score, cgr_cls,tracker_cgr)
    if isinstance(box, numpy.ndarray) and box.shape[0]!=0:
        boxes = xywh2xyxy_rescale(box, 0, False)
        return boxes,result
    else:
        return np.array([]),np.array([])