from roboflow import Roboflow

rf = Roboflow(api_key="sfoiasndao89")

project = rf.workspace("machli").project("fish-detection-model")

dataset = project.version(1)

dataset.download("yolov8")   # YOLO format (works for YOLOv5/v8/v11)
