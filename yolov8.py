import time
from ultralytics import YOLO
import cv2
from utils import detect_and_draw,cgr_detect
# Load a model

stream="rtmp://10.50.7.204:1935/live/stream"
cap= cv2.VideoCapture(stream)
cap.set(3, 1920)
cap.set(4, 1080)
cgr = YOLO("model/last.pt")
model = YOLO('model/yolov8n-pose.pt')
cv2.namedWindow("Multi Person Pose Detection",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Multi Person Pose Detection", 1920,1080)


if __name__ == '__main__':
    while cap.isOpened():
        start_time = time.time()
        ret, img = cap.read()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape!=(1920,1080):
            img = cv2.resize(img, (1920, 1080))
        # Predict with the model
        results = model.predict(img,device="cpu")
        # cgr_detect(results,img)

        res_plotted = detect_and_draw(results[0],img,cgr)
        # lock_hands(results[0], res_plotted)
        # res_plotted = plot_test(img,results[0])
        current_time = time.time()
        elapsed_time = current_time - start_time

        # 计算实时帧率
        fps = 1 / elapsed_time
        # 在帧上绘制实时帧率
        cv2.putText(res_plotted, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Multi Person Pose Detection", res_plotted)
        # print("done!")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # predict on an image