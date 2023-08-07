import cv2
from ultralytics import YOLO

model = YOLO('model/last.pt')
model.export(format='onnx',imgsz=(640,640))
