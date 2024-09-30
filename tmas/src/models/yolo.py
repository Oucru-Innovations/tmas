# tmas/tmas/models/yolo.py
import os
from ultralytics import YOLO
import os

# Get the directory of the current file
current_dir = os.path.dirname(__file__)
# print(f'YOLO file curent path: {current_dir} ')
# Construct the full path to the model file
weights = os.path.join(current_dir, 'best_model_yolo.pt')
# print(f'YOLO weight path: {weights} ')
class YOLOv8:
    def __init__(self, weights= weights):
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Model file {weights} not found. Please ensure the model file is in the correct path.")
        
        # Load the YOLOv8 model
        self.model = YOLO(weights)

    def predict(self, img):
        # Predict using the loaded model
        return self.model(img)
