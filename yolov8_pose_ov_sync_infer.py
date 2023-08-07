from openvino.runtime import Core
import numpy as np
import cv2, time
from yolov8_ByteTrack.infer import bytetrack

CLASSES={0: 'person'}
MODEL_NAME = "yolov8n-pose"
colors = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
          (255, 170, 0), (0, 255, 0), (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
          (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170, 255))


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_key_points(img, key_points, score_threshold, scale):
    key_points = np.array(key_points).reshape((-1, 3))
    for i, key_point in enumerate(key_points):
        points, score = key_point[0:2], key_point[2]
        if score > score_threshold:
            cv2.circle(img, tuple((points*scale).astype(int)), 5, colors[i], -1)


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0]
    y[..., 1] = x[..., 1]
    y[..., 2] = x[..., 0] + x[..., 2]
    y[..., 3] = x[..., 1] + x[..., 3]
    return y


# 实例化Core对象
core = Core() 
# 载入并编译模型
net = core.compile_model("model/yolov8n-pose.onnx", "AUTO")
# 获得模型输出节点
output_node = net.outputs[0]  
ir = net.create_infer_request()
cap = cv2.VideoCapture("smoke2.mp4")

while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
          break
    [height, width, _] = frame.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame
    scale = length / 640

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    # 基于OpenVINO实现推理计算
    outputs = ir.infer(blob)[output_node]
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]
    # Postprocess
    boxes = []
    scores = []
    preds_kpts = []
    for i in range(rows):
        classes_scores = outputs[0][i][4]
        key_points = outputs[0][i][5:]
        if classes_scores >= 0.5:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(classes_scores)
            preds_kpts.append(key_points)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []
    # for i in range(len(result_boxes)):
    #     index = result_boxes[i]
    box = np.array(boxes)[result_boxes].reshape(-1,4)
    box=xywh2xyxy(box)
    scores=np.array(scores)[result_boxes]
    result=bytetrack(box,scores)
    pred_kpts = np.array(preds_kpts)[result_boxes]

        # detection = {
        #     'class_id': 0,
        #     'class_name': 'person',
        #     'confidence': scores[index],
        #     'box': box,
        #     'scale': scale}
        # detections.append(detection)
        # print(box[0] * scale, box[1] * scale, scale)
        # draw_bounding_box(frame, 0, scores[index], round(box[0] * scale), round(box[1] * scale),
        #                   round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))
        # draw_key_points(frame, pred_kpts, 0.2, scale)

    end = time.time()
    # show FPS
    fps = (1 / (end - start))
    print(end-start)
    fps_label = "Throughput: %.2f FPS" % fps
    cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('YOLOv8 OpenVINO Infer Demo on AIxBoard', frame)
    # wait key for ending
    if cv2.waitKey(1) > -1:
        print("finished by user")
        break

cap.release()
cv2.destroyAllWindows()
