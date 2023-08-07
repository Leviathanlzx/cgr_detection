from ultralytics import YOLO

# Load a model
# build a new model from YAML

if __name__ == '__main__':
    model = YOLO("yolov8n-biformer.yaml").load("model/yolov8n.pt")  # load a pretrained model (recommended for training)
    # Train the model
    model.train(data='data.yaml', epochs=100)
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print(metrics.box.maps) # a li
