from datetime import datetime
import cv2
import numpy as np
from ov_inference import cgr_detect_with_onnx
# import trt_inference_detr
# import trt_inference_yolo

class Colors:
    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to rgb values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


# 绘制函数颜色库
colors = Colors()
kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

# 保存间隔
count = [0]
# 吸烟置信度
cgr_conf=[0.4]
# 使用模型
model=[0]
# 人员信息列表，包括每个人的id与吸烟检测累计值，累计超过吸烟设定阈值就会被认为在吸烟
ids = {}


def init_model(models):
    model[0]=models

def judge_smoke(pose_result, img, label):
    k = pose_result.keypoints
    left_angle, right_angle = cal_angle(k)
    left_hand_index = 9
    right_hand_index = 10
    # 如果角度小于55度或受手嘴距离小于0.8
    if int(left_angle) < 55 or cal_dis(k, left_hand_index) < 0.8:
        if cgr_detect(pose_result, img, left_hand_index, label):
            # 检测到香烟
            return 2
        else:
            return 1

    elif int(right_angle) < 55 or cal_dis(k, right_hand_index) < 0.8:
        if cgr_detect(pose_result, img, right_hand_index, label):
            return 2
        else:
            return 1

    return 0


def detect_and_draw(pose_result, img,opt):
    smoking_threshold=opt.threshold
    cgr_conf[0]=opt.cgr_conf
    cgrlabel = []
    # 画人物框
    for d in pose_result:
        # 每个人员的id与置信度
        conf, idd = float(d.conf), None if d.id is None else int(d.id)
        if idd not in ids.keys():
            # 加入人员追踪列表，idd为id，0为初始累计值
            ids[idd] = np.array([idd, 0])
        # name = ('' if id is None else f'id:{id} ')
        # label = (f'{name} {conf:.2f}' if conf else name)
        # if conf > 0.3:
        #     box_label(d.xyxy, img, lw, label)

        # 判断吸烟状态
        condition = ids[idd]
        status = judge_smoke(d, img, cgrlabel)
        # 吸烟状态阈值加10，不吸烟时每帧减少1，缓慢下降
        if status == 2:
            if condition[1] < 100:
                condition[1] += 10
            if condition[1] < smoking_threshold:
                box_label(d.xyxy, img, 3, "Suspicious", (28, 172, 255))
        elif status == 1:
            if condition[1] > 0:
                condition[1] -= 1
            box_label(d.xyxy, img, 3, "Suspicious", (28, 172, 255))
        else:
            if condition[1] > 0:
                condition[1] -= 1

        # 若超过设定阈值就显示
        if condition[1] > smoking_threshold:
            box_label(d.xyxy, img, 3, "Target is Smoking", (0, 0, 255))
        # 更新人员信息列表
        ids[idd] = condition
        # 画骨架
        if opt.skeleton:
            key_label(d.keypoints, img, img.shape, kpt_line=True)

    cgr_box = np.array([t[:4] for t in cgrlabel])
    # cgr_score = np.array([t[4] for t in cgrlabel])
    # cgr_box,cgr_score=cgr_update(cgr_box,cgr_score)
    # 画香烟
    if opt.cig_box:
        for i in cgr_box:
            box_label(i, img,3,label='Cig', color=(0, 0, 255), txt_color=(255, 255, 255))

    return img


def cal_dis(kpt, direction):
    # 计算手嘴距离，以上身长度为参照
    nose, wrist, shoulder, hip = kpt[0], kpt[direction], kpt[5], kpt[11]
    difference = nose - wrist
    standard = shoulder - hip
    # 计算欧氏距离
    distance = np.linalg.norm(difference)
    standdis = np.linalg.norm(standard)
    return distance / standdis


def cal_angle(kpt):
    # 计算关节夹角，以向量夹角方式计算
    lshoulder, lelbow, lwrist = kpt[5], kpt[7], kpt[9]
    rshoulder, relbow, rwrist = kpt[6], kpt[8], kpt[10]
    left_shoulder_vector = lshoulder - lelbow
    left_wrist_vector = lwrist - lelbow
    right_shoulder_vector = rshoulder - relbow
    right_wrist_vector = rwrist - relbow
    # 计算向量的夹角（弧度）
    left_angle_radian = np.arccos(
        np.dot(left_shoulder_vector, left_wrist_vector) / (
                np.linalg.norm(left_shoulder_vector) * np.linalg.norm(left_wrist_vector)))
    right_angle_radian = np.arccos(
        np.dot(right_shoulder_vector, right_wrist_vector) / (
                np.linalg.norm(right_shoulder_vector) * np.linalg.norm(right_wrist_vector)))

    # 将弧度转换为角度
    right_angle_degree = np.degrees(right_angle_radian)
    left_angle_degree = np.degrees(left_angle_radian)

    return left_angle_degree, right_angle_degree


def box_label(box, im, lw, label='', color=(255, 255, 64), txt_color=(255, 255, 255)):
    # 画出检测框，box为xyxy格式，来源于ultralytics官方
    """Add one xyxy box to image with label."""
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)


def cgr_detect(k, img, direction, label):
    count[0] += 1
    box = k.xyxy
    right = k.keypoints[0]
    # 锁定嘴部位置
    length = int(0.4 * (box[2] - box[0]))
    lengths = int(0.3 * (box[3] - box[1]))
    box = box.astype(np.int32)
    box[1] = np.max([int(right[1]) - length, 0])
    box[3] = np.min([int(right[1]) + length, img.shape[0]])
    box[0] = np.max([int(right[0]) - lengths, 0])
    box[2] = np.min([int(right[0]) + lengths, img.shape[1]])
    # 挖出嘴部图片
    person = img[box[1]:box[3], box[0]:box[2]]
    # if count[0] % 5 == 0:
    #     cv2.imwrite(f"video/{k.id}.jpg", person)

    if person.shape[0] != 0 and person.shape[1] != 0:
        # 对挖出图片进行香烟目标检测
        if model[0]==0:
            boxes, scores = trt_inference_detr.cgr_detect_with_onnx(person)
        if model[0]==1:
            boxes, scores = trt_inference_yolo.cgr_detect_with_onnx(person)
        if model[0]==2:
            boxes, scores = cgr_detect_with_onnx(person)
        # boxes, scores = cgr_detect_alternative(person)
        for i, c in enumerate(scores):
            # 若存在，则添加至香烟队列（用于画图）
            if c > cgr_conf[0]:
                label.append([int(boxes[i][0]) + int(box[0]), int(boxes[i][1]) + int(box[1]),
                              int(boxes[i][2]) + int(box[0]), int(boxes[i][3]) + int(box[1]), c])
                return True
            else:
                return False


def key_label(kpts, im, shape=(640, 640), radius=5, kpt_line=True):
    # 骨架绘图函数，来源于ultralytics库
    """
    Args:
        kpts (ndarray): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
        shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
        radius (int, optional): Radius of the drawn keypoints. Default is 5.
        kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                   for human pose. Default is True.
    """
    nkpt, ndim = kpts.shape
    is_pose = nkpt == 17 and ndim == 3
    kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
    for i, k in enumerate(kpts):
        color_k = [int(x) for x in kpt_color[i]]
        x_coord, y_coord = k[0], k[1]
        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
            if len(k) == 3:
                conf = k[2]
                if conf < 0.4:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

    if kpt_line:
        ndim = kpts.shape[-1]
        for i, sk in enumerate(skeleton):
            pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
            pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
            if ndim == 3:
                conf1 = kpts[(sk[0] - 1), 2]
                conf2 = kpts[(sk[1] - 1), 2]
                if conf1 < 0.5 or conf2 < 0.5:
                    continue
            if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                continue
            if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                continue
            cv2.line(im, pos1, pos2, [int(x) for x in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

