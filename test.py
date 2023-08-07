import onnxruntime as rt
import cv2
import numpy as np


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



sess=rt.InferenceSession("model/rtdetr-best.onnx")
img=cv2.imread("video/105.jpg")
images=cv2.imread("video/105.jpg")

[height, width, _] = img.shape
length = max((height, width))
image = np.zeros((length, length, 3), np.uint8)
image[0:height, 0:width] = img
scale = length / 640
image = cv2.resize(image, (640, 640))
image = image.astype(np.float32) / 255.0
input_img = np.transpose(image, [2, 0, 1])
image= input_img[np.newaxis, :, :, :]


# print(img.shape)
# org_img=img
# im_shape=np.array([[float(img.shape[0]),float(img.shape[1])]]).astype('float32')
# img=cv2.resize(img,(640,640))
# scale_factor=np.array([[float(640/img.shape[0]),float(640/img.shape[1])]]).astype('float32')
# img=img.astype(np.float32)/255.0
# input_img=np.transpose(img,[2,0,1])
# image=input_img[np.newaxis,:,:,:]
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
result=sess.run([label_name],{input_name:image})
# outputs = np.transpose(result[0].squeeze(),[1,2,0])*255
# print(outputs.shape)
boxs,score=postprocess(result[0].squeeze(),0.5,(640,640))
print(boxs)

# cv2.imwrite("video/result.jpg",outputs)
    # if value[1]>0.
for box in boxs:
    cv2.rectangle(images,(int(box[0]*scale),int(box[1]*scale)),(int(box[2]*scale),int(box[3]*scale)),(255,0,0),2)
cv2.imshow("te",images)
cv2.waitKey(0)
    #     cv2.putText(org_img,str(int(value[0]))+":"+str(value[1]),(int(value[2]),int(value[3])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    #     cv2.imwrite("result.png",org_img)