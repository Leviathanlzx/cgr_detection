import cv2
import numpy as np
from typing import Tuple
import openvino.runtime as ov
from onnx_inference import *

core = ov.Core()
compiled_model = core.compile_model("models/yolov8n-pose.onnx", "AUTO")
infer_request = compiled_model.create_infer_request()

img=cv2.imread("test/2023-07-24 16-54-16.jpg")
img,border=prepare_input(img)
# image = cv2.imread("hands/220.jpg")
# Create tensor from external memory
input_tensor = ov.Tensor(array=img)
# Set input tensor for models with one input
infer_request.set_input_tensor(input_tensor)
infer_request.infer()
# infer_request.start_async()
# infer_request.wait()
# Get output tensor for models with one output
output = infer_request.get_output_tensor()
output_buffer = output.data
print(output_buffer.shape)