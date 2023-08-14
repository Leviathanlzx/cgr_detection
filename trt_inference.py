import time
import cv2
import numpy
import numpy as np
from typing import Tuple
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from bytetrack_init import bytetrack, make_parser
from yolov8onnx.utils import xywh2xyxy, nms
from ultralytics.trackers import BYTETracker
from test import SR

f1 = open("model/yolov8x-pose.trt", "rb")
f2 = open("model/detr.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine1 = runtime.deserialize_cuda_engine(f1.read())
context1 = engine1.create_execution_context()
engine2 = runtime.deserialize_cuda_engine(f2.read())
context2 = engine2.create_execution_context()

img = np.zeros([1, 3, 640, 640])
output1 = np.empty([1, 56, 8400], dtype=np.float32)
d_input1 = cuda.mem_alloc(1 * img.nbytes)
d_output1 = cuda.mem_alloc(1 * output1.nbytes)

bindings1 = [int(d_input1), int(d_output1)]

output2 = np.empty([1, 300, 5], dtype=np.float32)
d_input2 = cuda.mem_alloc(1 * img.nbytes)
d_output2 = cuda.mem_alloc(1 * output2.nbytes)

bindings2 = [int(d_input2), int(d_output2)]
stream = cuda.Stream()

args = make_parser().parse_args()
tracker = BYTETracker(args, frame_rate=30)
tracker_cgr = BYTETracker(args, frame_rate=30)


def predict(stream, bindings, batch, context, d_input, d_output, output):  # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()

    return output


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

    return boxes, scores


def xywh2xyxy_rescale(x, scale, is_scale):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    if is_scale:
        y[..., 0] = x[..., 0] * scale
        y[..., 1] = x[..., 1] * scale
        y[..., 2] = (x[..., 0] + x[..., 2]) * scale
        y[..., 3] = (x[..., 1] + x[..., 3]) * scale
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
    blob = np.ascontiguousarray(input_img, dtype=np.float32)
    # Create tensor from external memory
    # blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    # blob=blob.astype(np.float16)
    start_time = time.time()
    outputs = predict(stream, bindings2, blob, context2, d_input2, d_output2, output2)
    current_time = time.time()
    elapsed_time = current_time - start_time
    # print("done!", elapsed_time)
    # outputs = np.transpose(outputs[0],[0,2,1])
    length /= 3
    boxes, scores = postprocess(outputs.squeeze(), 0.35, (length, length))
    # boxes, result = bytetrack(boxes, scores,class_ids, tracker_cgr)
    if isinstance(boxes, numpy.ndarray) and boxes.shape[0] != 0:
        # boxes = xywh2xyxy_rescale(boxes, 0, False)
        return boxes, scores
    else:
        return [], []


def pose_estimate_with_onnx(frame):
    start_time = time.time()
    [height, width, _] = frame.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame

    image = cv2.resize(image, (640, 640))
    scale = length / 640
    image = image.astype(np.float32) / 255.0
    image = image[:, :, ::-1]
    input_img = np.transpose(image, [2, 0, 1])
    blob = input_img[np.newaxis, :, :, :]
    blob = np.ascontiguousarray(blob, dtype=np.float32)
    # blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    # 基于tensorRT实现推理计算

    outputs = predict(stream, bindings1, blob, context1, d_input1, d_output1, output1)
    outputs = np.transpose(outputs, [0, 2, 1])

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
    box = xywh2xyxy_rescale(box, scale, True)
    scores = np.array(scores)[result_boxes]
    clss = numpy.zeros(box.shape[0])
    box, result = bytetrack(box, scores, clss, tracker)
    if isinstance(box, numpy.ndarray) and box.shape[0] != 0:
        box = xywh2xyxy_rescale(box, scale, False)
    kpts = np.array(preds_kpts)[result_boxes].reshape(-1, 17, 3) * scale

    return box, scores, result, kpts


def cgr_update(box, score):
    cgr_cls = numpy.zeros(box.shape[0])
    box, result = bytetrack(box, score, cgr_cls, tracker_cgr)
    if isinstance(box, numpy.ndarray) or box.shape[0] != 0:
        boxes = xywh2xyxy_rescale(box, 0, False)
        return boxes, result
    else:
        return np.array([]), np.array([])