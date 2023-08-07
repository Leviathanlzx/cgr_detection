import time
from ultralytics import YOLO,RTDETR
import cv2
from utils import detect_and_draw,cgr_detect
# Load a model


model = RTDETR('model/rtdetr.pt')
cv2.namedWindow("Multi Person Pose Detection",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Multi Person Pose Detection", 1920,1080)


if __name__ == '__main__':
    img=cv2.imread("video/result.jpg")
    results = model.predict(img,device="cpu")[0].plot()
    # cgr_detect(results,img)
    cv2.imshow("Multi Person Pose Detection",results)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # predict on an image