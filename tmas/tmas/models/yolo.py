# tmas/tmas/models/yolo.py
import os
from ultralytics import YOLO
class YOLOv8:
    def __init__(self, weights="tmas/tmas/models/best_model_yolo.pt"):
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Model file {weights} not found. Please ensure the model file is in the correct path.")
        
        # Load the YOLOv8 model
        self.model = YOLO(weights)

    def predict(self, img):
        # Predict using the loaded model
        return self.model(img)
