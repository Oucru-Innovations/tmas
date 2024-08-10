# src/tmas/models/yolo.py
from ultralytics import YOLO

class YOLOv8:
    def __init__(self, weights='models/best_model_yolo.pt'):
        # Load the YOLOv8 model directly using the ultralytics package
        self.model = YOLO(weights)
    
    def predict(self, image):
        # Perform detection
        results = self.model(image)
        return results