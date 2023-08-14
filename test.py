import onnxruntime as rt
import cv2
import numpy as np

sess=rt.InferenceSession("model/SR.onnx",providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name



def SR(img):
# img=cv2.imread("video/1955.jpg")
# [height, width, _] = img.shape
# length = max((height, width))
# image = np.zeros((length, length, 3), np.uint8)
# image[0:height, 0:width] = img
# scale = length / 640
# image = cv2.resize(image, (640, 640))
    image = img.astype(np.float32)/255
    input_img = np.transpose(image, [2, 0, 1])
    image= input_img[np.newaxis, :, :, :]


# print(img.shape)
# org_img=img
# im_shape=np.array([[float(img.shape[0]),float(img.shape[1])]]).astype('float32')
# img=cv2.resize(img,(640,640))
# scale_factor=np.array([[float(640/img.shape[0]),float(640/img.shape[1])]]).astype('float32')
# img=img.astype(np.float32)/255.0
# input_img=np.transpose(img,[2,0,1])
# image=input_img[np.newaxis,:,:,:]
    result=sess.run([label_name],{input_name:image})
    # result=sess.run(None,{"inputs":image,"scale":np.array(4).astype(np.int64)})
    outputs = np.transpose(result[0].squeeze(),[1,2,0])*255
    return outputs
# boxs,score=postprocess(result[0].squeeze(),0.5,(640,640))
# print(boxs)

# cv2.imwrite("video/result.jpg",outputs*255)
    # if value[1]>0.
# for box in boxs:
#     cv2.rectangle(images,(int(box[0]*scale),int(box[1]*scale)),(int(box[2]*scale),int(box[3]*scale)),(255,0,0),2)
# cv2.imshow("te",images)
# cv2.waitKey(0)
    #     cv2.putText(org_img,str(int(value[0]))+":"+str(value[1]),(int(value[2]),int(value[3])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    #     cv2.imwrite("result.png",org_img)