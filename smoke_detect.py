from ultralytics import YOLO
import cv2


cap= cv2.VideoCapture("smoke2.mp4")
cv2.namedWindow("Multi Person Pose Detection",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Multi Person Pose Detection", 1280, 720)

model = YOLO('best.pt')
while cap.isOpened():
    ret, img = cap.read()
    results = model.predict(img,device="cpu")
    res_plotted = results[0].plot()
    cv2.imshow("Multi Person Pose Detection", res_plotted)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()