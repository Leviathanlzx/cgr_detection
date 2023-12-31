import time
from ultralytics import YOLO,RTDETR
import cv2
from utils import detect_and_draw,cgr_detect
# Load a models


model = RTDETR('model/rtdetr.pt')
cv2.namedWindow("Multi Person Pose Detection",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Multi Person Pose Detection", 1920,1080)


if __name__ == '__main__':
    img=cv2.imread("video/result.jpg")
    results = model.predict(img,device="cpu",conf=0.1)[0]
    print(results.boxes.conf)
    # cgr_detect(results,img)
    cv2.imshow("Multi Person Pose Detection",results.plot())
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # predict on an image