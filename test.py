import onnxruntime as rt
import cv2
import numpy as np

sess=rt.InferenceSession("model/yolov8n-pose.onnx")
img=cv2.imread("video/smoker.jpg")
org_img=img
im_shape=np.array([[float(img.shape[0]),float(img.shape[1])]]).astype('float32')
img=cv2.resize(img,(640,640))
scale_factor=np.array([[float(640/img.shape[0]),float(640/img.shape[1])]]).astype('float32')
img=img.astype(np.float32)/255.0
input_img=np.transpose(img,[2,0,1])
image=input_img[np.newaxis,:,:,:]
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
result=sess.run([label_name],{input_name:image})
outputs = np.transpose(result[0],[0,2,1])
print(outputs.shape)
    # if value[1]>0.5:
    #     cv2.rectangle(org_img,(int(value[2]),int(value[3])),(int(value[4]),int(value[5])),(255,0,0),2)
    #     cv2.putText(org_img,str(int(value[0]))+":"+str(value[1]),(int(value[2]),int(value[3])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    #     cv2.imwrite("result.png",org_img)