from datetime import datetime
import cv2
import numpy as np
import torch
from ort_inference import cgr_detect_with_onnx


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


colors = Colors()
kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                         [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
count=[0]
smoking_threshold = 5  # 吸烟检测阈值，连续检测到吸烟的次数
no_smoking_threshold = 10 # 非吸烟检测阈值，连续未检测到吸烟的次数
ids={}

def judge_smoke(pose_result,img):
    """
    判断是否出现吸烟行为。

    参数:
    pose_result (PoseResult): 姿态估计的结果，包含人体关键点信息。
    img (Image): 输入图像，用于传递给 cgr_detect 函数进行烟雾检测。
    cgr (CGR): cgr_detect 函数所需的其他参数。

    返回值:
    0: 未检测到吸烟行为。
    1: 检测到单手吸烟行为。
    2: 检测到双手吸烟行为。
    """
    k = pose_result.keypoints
    left_angle, right_angle = cal_angle(k)
    left_hand_index = 9
    right_hand_index = 10

    if int(left_angle) < 40 or cal_dis(k, left_hand_index)<0.8:
        if cgr_detect(pose_result, img, left_hand_index):
            return 2
        else:
            return 1

    elif int(right_angle) < 40 or cal_dis(k, left_hand_index)<0.8:
        if cgr_detect(pose_result, img, right_hand_index):
            return 2
        else:
            return 1

    return 0


def detect_and_draw(pose_result, img):
    lw = max(round(sum(img.shape) / 2 * 0.003), 2) # 线宽

    # 画人物框
    for d in pose_result:
        conf, idd =float(d.conf), None if d.id is None else int(d.id)
        if idd not in ids.keys():
            ids[idd]=np.array([idd,0,0,False])
        # name = ('' if id is None else f'id:{id} ')
        # label = (f'{name} {conf:.2f}' if conf else name)
        # if conf > 0.3:
        #     box_label(d.xyxy, img, lw, label)

        # 时长判断
        condition = ids[idd]
        status = judge_smoke(d, img)
        if status == 2:
            if condition[1] < smoking_threshold:
                box_label(d.xyxy, img, 3, str(idd) + "suspicious", (28, 172, 255))
            condition[1] += 1
            if condition[2] > 0:
                condition[2] -= 1
        elif status == 1:
            box_label(d.xyxy, img, 3, str(idd) + "suspicious", (28, 172, 255))
        else:
            condition[2] += 1
            if condition[1] > 0:
                condition[1] -= 1
        # 根据吸烟检测阈值和非吸烟检测阈值进行判断
        if condition[1] >= smoking_threshold:
            condition[3] = True
            condition[2] = 0
            # print(f"{id} 吸烟")
            box_label(d.xyxy, img, 3, "ID" + str(idd) + " smoking", (0, 0, 255))

        elif condition[2] >= no_smoking_threshold:
            condition[3] = False

        ids[idd] = condition
            # print(f"{id} 没有吸烟")
        # cv2.putText(img, str(float(dist)),(int(x_coord), int(y_coord)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.circle(img, (int(x_coord), int(y_coord)), 5, (255, 255, 0), -1, lineType=cv2.LINE_AA)
    
    # 画骨架
    for k in pose_result:
        key_label(k.keypoints, img,img.shape, kpt_line=True)

    return img


def cal_dis(kpt,direction):
    nose,wrist,shoulder,hip =kpt[0],kpt[direction],kpt[5],kpt[11]
    difference = nose - wrist
    standard = shoulder-hip
    # 计算欧氏距离
    distance = np.linalg.norm(difference)
    standdis= np.linalg.norm(standard)
    return distance/standdis


def cal_angle(kpt):
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

    return left_angle_degree,right_angle_degree


def box_label(box, im, lw, label='', color=(255,255,64), txt_color=(255, 255, 255)):
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


def cgr_detect(k, img,direction):
    count[0] += 1
    box = k.xyxy
    right = k.keypoints[0]
    # print("there")
    length=int(0.5*(box[2]-box[0]))
    lengths=int(0.4*(box[3]-box[1]))
    box=box.astype(np.int32)
    box[1] = np.max([int(right[1]) - length,0])
    box[3] = np.min([int(right[1]) + length,img.shape[0]])
    box[0] = np.max([int(right[0]) - lengths,0])
    box[2] = np.min([int(right[0]) + lengths,img.shape[1]])
    person = img[box[1]:box[3],box[0]:box[2]]
    # cv2.imshow("hands",person)
    # cv2.waitKey(5)
    # if count[0]%5==0:
    #     cv2.imwrite(f"hands/{count[0]}.jpg",person)

    if person.shape[0] != 0 and person.shape[1] != 0:
        boxes, scores=cgr_detect_with_onnx(person)
        # boxes, scores = cgr_detect_alternative(person)
        for i, c in enumerate(scores):
            if c > 0.4:
                cgr_label(boxes[i], box, img)
                if count[0] % 10 == 0:
                    bo = k.xyxy
                    t = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
                    cv2.imwrite(f"test/{t}.jpg",img[int(bo[1]):int(bo[3]), int(bo[0]):int(bo[2])])
                    # print("finish!")
                return True
            else:
                return False
    # cgr_results = cgr.predict(person, imgsz=640)
    # for i, c in enumerate(cgr_results[0].boxes):
    #     conf = float(c.conf)
    #     if conf > 0.4:
    #         t = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    #         # cgr_label(c.xyxy.squeeze(), box, img)
    #         if count[0]%10==0:
    #             bo = k.boxes.xyxy.squeeze().tolist()
    #             cv2.imwrite(f"test/{t}.jpg",img[int(bo[1]):int(bo[3]), int(bo[0]):int(bo[2])])
    #             print("finish!")
    #         return True

    # except:
    #     person = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]



def cgr_label(box, ori, im, label='Cig', color=(0, 0, 255), txt_color=(255, 255, 255)):
    """Add one xyxy box to image with label."""
    # lw = max(round(sum(im.shape) / 2 * 0.003), 2)
    lw=3
    if isinstance(box, torch.Tensor):
        box = box.tolist()
    p1, p2 = (int(box[0]) + int(ori[0]), int(box[1]) + int(ori[1])), (
    int(box[2]) + int(ori[0]), int(box[3]) + int(ori[1]))
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


def key_label(kpts,im, shape=(640, 640), radius=5, kpt_line=True):
    """Plot keypoints on the image.

    Args:
        kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
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


# def lock_hands(pose_result,img,direction):
#     for i, k in enumerate(pose_result):
#         right=k.keypoints.xy[0][direction]
#         box=k.boxes.xyxy.squeeze().tolist()
#         if right[0]+32>int(box[2]):
#             br=right[0]
#
#         # cv2.circle(img, (int(right[0]), int(right[1])), 5, (255,255,255), -1, lineType=cv2.LINE_AA)
#         try:
#             if count[0]%10==0:
#                 cv2.imwrite(f"hands/{count[0]}.jpg",img[int(right[1])-64:int(right[1])+64,int(right[0])-32:int(right[0])+32])
#             return img[int(right[1])-64:int(right[1])+64,int(right[0])-32:int(right[0])+32]
#         except:
#             return img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

