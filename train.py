from ultralytics import YOLO

# Load a models
# build a new models from YAML

if __name__ == '__main__':
    model = YOLO("yolov8n-biformer.yaml").load("models/yolov8n.pt")  # load a pretrained models (recommended for training)
    # Train the models
    model.train(data='data.yaml', epochs=100)
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print(metrics.box.maps) # a li
