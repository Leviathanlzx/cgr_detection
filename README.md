# 吸烟检测 Cigarette Detection
YOLOv8-Pose+BYTETRACK+RTDETR Cigarette Detection+Openvino/Onnxruntime/TensorRT部署
## 具体流程 Process

>YOLOv8-pose首先检测图片中人体位置并抽取骨架信息，根据人体骨架姿态判断每个人的手肘弯曲程度以及手与嘴部距离；若判定成功，使用RTDETR对人嘴部位置进行香烟目标检测；若检出目标，则增加对该人的吸烟累计值，当累计值超过一定阈值后，判断为正在吸烟（连续时间判断，减少单帧判断造成的结果不稳定与不准确）。同时，如一段时间检测不到香烟，则累计值缓慢下降至阈值下（停止判断为吸烟状态）
><br>为了对目标进行累计值计算，使用bytetrack算法追踪每个人体目标，保证每个人的ID不变

The YOLOv8-pose project begins by detecting human body positions and extracting skeletal information from images. Based on the skeletal poses, it assesses the elbow angles and the distance between hands and mouths for each individual. If successful, the RTDETR model is employed to detect cigarettes at the mouth position.

Upon target detection, the cumulative smoking count for that individual increases. Smoking is determined when the cumulative count surpasses a predefined threshold, ensuring continuous judgment over time to minimize unstable and inaccurate single-frame assessments. In the absence of cigarette detection for a period, the cumulative count gradually decreases until reaching a lower threshold, indicating the cessation of smoking judgment.

To facilitate cumulative calculations, the bytetrack algorithm is used to track each individual, ensuring consistent identification by preserving unique IDs for each person. This project aims to provide a reliable system for real-time monitoring of smoking behavior through skeletal and object detection.

## 代码说明
### 模型使用

>整体来看，现有的目标检测框架大致可以分为CNN based以及Transformer based。对于前者，通常又可以划分为以Faster RCNN和RetinaNet为代表的“学术派”和以YOLO系列为代表的“工业派”。但作为检测领域的另一个巨头——DETR系列，相关研究很少会涉及到“实用性”，大多数都还是在验证新模块、新改进和新优化的“可行性”。对于这个问题，百度近期（2023.4）提交了一份“答案”：[RT-DETR](https://arxiv.org/abs/2304.08069)。百度之所以做这么一件事，其目的是希望为工业界提供一款实用性较高的DETR系列的实时检测器。相较于最新的YOLOv8，RT-DETR以较短的训练时长（75~80 epoch）和较少的数据增强（没有马赛克增强）的策略，在同等测试条件下（640x640）展现出了更强的性能和更好的平衡，且检测速度也与YOLO系列相媲美。
>
>通过使用相同的数据集分别在YOLOv8n、YOLOv8s、YOLOv8m以及RTDETR-l上训练，发现RTDETR在模型性能上完全超越了YOLO系列，达到了0.95099 precision、0.92931	recall、0.9612 mAP50、0.61979 mAP50-95。因此，选用RTDETR作为香烟检测模型
>
>相对的，目前使用的RTDETR网络规模较大(120MB+)，如遇到算力不足问题，可重新训练较小的模型(可使用ultralytics训练，需要修改模型的YAML来修改网络规模)
### 项目说明
    │  bytetrack_init.py                  bytrack初始化与参数设定
    │  byte_tracker.py                    替换ultralytics库下的trackers/byte_tracker.py
    │  export.py                          从pt模型导出onnx模型
    │  func.py                            程序逻辑处理主要函数，如吸烟判断函数
    │  infer_main.py                      主程序，启动推理
    │  ort_inference.py                   onnxruntime推理模块
    │  ov_inference.py                    openvino推理模块    
    │  qt_main.py                         GUI窗口启动
    │  requirements.txt                   项目环境需求
    │  trt_inference_detr.py              detr-tensorrt推理
    │  trt_inference_yolo.py              yolo-tensorrt推理
    │  un.ui                              qt的ui界面文件
    │  trt_infer.ipynb                    从onnx模型生成本机的trt模型
    ├─model                               模型文件夹
    │      rtdetr-best.onnx               rtdetr的的onnx模型
    │      rtdetr-best.pt                 rtdetr的的PT模型，可使用ultralytics库的RTDETR载入
    │      rtdetr-best.trt                rtdetr的的trt模型(测试用,请重新生成)
    │      yolov8m-pose.onnx              yolov8-pose的onnx模型
    │      yolov8m-pose.trt               yolov8-pose的trt模型(测试用,请重新生成)
    │      yolov8n-cig.onnx               yolov8的onnx模型
    │      yolov8n-pose.onnx              yolov8-pose的onnx模型
    │      yolov8n-cig.pt                 yolov8的pt模型
    ├─trtmodel                            存放新的trt模型
    ├─datasets                            训练用的香烟数据集
    │  workflow.png                       程序工作流程图
    │  funcall.png                        函数调用关系图
>更换推理框架：请修改func.py与infer_main.py中导入函数pose_estimate_with_onnx与cgr_detect_with_onnx的导入方式，可以选择ort_inference.py/ov_inference.py/trt_inference.py来导入上述函数，从而使用不同推理框架
> 
>文件替换：由于ultralytics库原生追踪算法与其自己的推理绑定，因此修改了部分库代码，请用项目内文件byte_tracker.py 替换ultralytics库内trackers/byte_tracker.py来使用bytetrack追踪

## 使用说明

运行以下代码启动GUI窗口：

    python qt_main.py

或运行以下代码直接进行推理:

    python infer_main.py
    
本程序默认为使用trt推理，因为rtdetr对于trt优化有着较大依赖（cpu推理约2000ms，gpu-ort推理约100ms，而gpu-trt推理可以加速到7ms，gpu为3090）
打开GUI后，首先选择使用模型进行初始化，之后即可正常操作

GUI可调整香烟框、骨架、香烟置信度、检测阈值等（不建议修改这个），并支持保存（应在开始推理前勾选）
点击“开始/继续”开始推理，界面实时显示推理效果

测试视频位于video文件夹内

### 注意事项
在新设备使用时，应先使用trt_infer.ipynb按步骤生成每个onnx模型在本机对应的trt模型；
<br>本程序已在3090服务器上使用tensorRT完成部署测试，但由于该服务器的pycharm不兼容某些32位的cuda dll，因此无法从pycharm启动（原因未知）
<br>正确启动方式为：打开conda prompt，输入：

    cd C:\Users\Administrator\Desktop\cgr\cgr_detection

然后启动conda虚拟环境cgr，输入：

    conda activate cgr

进入虚拟环境后通过控制台启动程序：

    python qt_main.py

显示GUI界面后，首先选择检测模型，并点击初始化模型，之后即可正常使用


