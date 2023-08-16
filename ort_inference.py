import time
import cv2
import numpy
import numpy as np
import onnxruntime as rt
from bytetrack_init import bytetrack,make_parser
from ultralytics.trackers import BYTETracker
# from test import SR

cgr_model=rt.InferenceSession("model/rtdetr-best.onnx")
input_name = cgr_model.get_inputs()[0].name
label_name = cgr_model.get_outputs()[0].name
pose_model=rt.InferenceSession("model/yolov8n-pose.onnx")
input_names = pose_model.get_inputs()[0].name
label_names = pose_model.get_outputs()[0].name
args = make_parser().parse_args()
tracker = BYTETracker(args, frame_rate=30)
tracker_cgr=BYTETracker(args, frame_rate=60)


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


def cgr_detect_with_onnx(frame):
    # frame = SR(frame)
    [height, width, _] = frame.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame
    scale = length / 640
    image = cv2.resize(image, (640, 640))
    image = image.astype(np.float32) / 255.0
    image = image[:, :, ::-1]
    input_img = np.transpose(image, [2, 0, 1])
    input_img = input_img[np.newaxis, :, :, :]
    # Create tensor from external memory
    outputs = cgr_model.run([label_name], {input_name: input_img})
    output_buffer=np.transpose(outputs[0],[0,2,1])
    boxes, scores= postprocess(outputs[0].squeeze(),0.25,(length,length))
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
    image = image[:, :, ::-1]
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