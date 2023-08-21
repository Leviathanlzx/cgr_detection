# 吸烟检测
RTDETR+BYTETRACK+YOLOv8吸烟检测+Openvino/Onnxruntime/TensorRT部署
## 具体流程

>YOLOv8-pose首先检测图片中人体位置并抽取骨架信息，根据人体骨架姿态判断每个人的手肘弯曲程度以及手与嘴部距离；若判定成功，使用RTDETR对人嘴部位置进行香烟目标检测；若检出目标，则增加对该人的吸烟累计值，当累计值超过一定阈值后，判断为正在吸烟（连续时间判断，减少单帧判断造成的结果不稳定与不准确）。同时，如一段时间检测不到香烟，则累计值缓慢下降至阈值下（停止判断为吸烟状态）

## 代码说明
### 模型使用

>整体来看，现有的目标检测框架大致可以分为CNN based以及Transformer based。对于前者，通常又可以划分为以Faster RCNN和RetinaNet为代表的“学术派”和以YOLO系列为代表的“工业派”。但作为检测领域的另一个巨头——DETR系列，相关研究很少会涉及到“实用性”，大多数都还是在验证新模块、新改进和新优化的“可行性”。对于这个问题，百度近期（2023.4）提交了一份“答案”：[RT-DETR](https://arxiv.org/abs/2304.08069)。百度之所以做这么一件事，其目的是希望为工业界提供一款实用性较高的DETR系列的实时检测器。相较于最新的YOLOv8，RT-DETR以较短的训练时长（75~80 epoch）和较少的数据增强（没有马赛克增强）的策略，在同等测试条件下（640x640）展现出了更强的性能和更好的平衡，且检测速度也与YOLO系列相媲美。
>通过使用相同的数据集分别在YOLOv8n、YOLOv8s、YOLOv8m以及RTDETR-l上训练，发现RTDETR在模型性能上完全超越了YOLO系列，达到了0.95099 precision、0.92931	recall、0.9612 mAP50、0.61979 mAP50-95。因此，选用RTDETR作为香烟检测模型

### 代码说明
    │  bytetrack_init.py  bytrack初始化与参数调整
    │  byte_tracker.py    替换ultralytics库下的trackers/byte_tracker.py
    │  export.py          导出onnx模型
    │  func.py            程序逻辑处理函数，如吸烟判断函数
    │  infer_main.py      主程序，启动推理
    │  LICENSE
    │  ort_inference.py   onnxruntime推理
    │  ov_inference.py    openvino推理    
    │  qt_main.py         GUI窗口启动
    │  requirements.txt   项目环境需求
    │  trt_inference_detr.py  detr-tensorrt推理
    │  trt_inference_yolo.py  yolo-tensorrt推理
    │  un.ui              qt的ui界面文件
    ├─model               模型文件夹
    │      last.onnx      yolov8的onnx模型
    │      rtdetr-best.onnx   rtdetr的的onnx模型
    │      rtdetr-best.pt    rtdetr的的PT模型，可使用ultralytics库的RTDETR载入
    │      rtdetr-best.trt   rtdetr的的trt模型
    │      yolov8m-pose.onnx    yolov8-pose的onnx模型
    │      yolov8m-pose.trt    yolov8-pose的trt模型
    │      yolov8n-cig.onnx    yolov8的onnx模型
    │      yolov8n-pose.onnx    yolov8-pose的onnx模型
    
>更换推理框架：请修改func.py与infer_main.py中导入函数pose_estimate_with_onnx与cgr_detect_with_onnx的导入方式，可以选择ort_inference.py/ov_inference.py/trt_inference.py来导入上述函数，从而使用不同推理框架

>请用项目内文件替换ultralytics库内trackers/byte_tracker.py以使用bytetrack追踪
