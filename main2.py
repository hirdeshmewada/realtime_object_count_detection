from ultralytics import YOLO
model = YOLO("yolov8s.pt")
print(model.model.names)  # Print the class name