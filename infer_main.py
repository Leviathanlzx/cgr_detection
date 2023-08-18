import cv2
import time
from ov_inference import pose_estimate_with_onnx
from func import detect_and_draw


class Result:
    def __init__(self,box,score,id,kpts):
        self.xyxy=box
        self.conf=score
        self.id=id
        self.keypoints=kpts


def mp4save():
    frame_width = 1920  # 输出视频宽度
    frame_height = 1080  # 输出视频高度
    fps = 30  # 帧率
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编码格式
    out = cv2.VideoWriter("video/output.mp4", fourcc, fps, (frame_width, frame_height))
    return out


def cgr_detect(image,opt):
    start_time = time.time()
    # 推理人体姿态
    box, score, ids, kpts = pose_estimate_with_onnx(image)
    if ids and box is not None:
        pose_result = [Result(i, j, k, m) for i, j, k, m in zip(box, score, ids, kpts)]
        # 检测吸烟
        image = detect_and_draw(pose_result, image,opt)

    current_time = time.time()
    elapsed_time = current_time - start_time
    # 计算实时帧率
    fps = 1 / elapsed_time
    # 在帧上绘制实时帧率
    cv2.putText(image, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image,fps,elapsed_time


if __name__ == '__main__':
    stream = "rtmp://10.50.7.204:1935/live/stream"
    cap = cv2.VideoCapture("video/test3.mp4")
    # cap.set(3, 3840)
    # cap.set(4, 2160)
    cv2.namedWindow("Multi Person Pose Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi Person Pose Detection", 1920, 1080)
    out = mp4save()

    while cap.isOpened():
        start_time = time.time()
        ret,image=cap.read()
        if not ret:
            break

        box, score, ids, kpts=pose_estimate_with_onnx(image)
        if ids and box is not None:
            pose_result=[Result(i,j,k,m) for i,j,k,m in zip(box, score, ids, kpts)]
            image = detect_and_draw(pose_result, image)

        current_time = time.time()
        elapsed_time = current_time - start_time
        # 计算实时帧率
        fps = 1 / elapsed_time
        # 在帧上绘制实时帧率

        cv2.putText(image, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Multi Person Pose Detection", image)
        out.write(image)
        # print("done!")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()