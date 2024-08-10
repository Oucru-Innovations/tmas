# tmas/detection.py
from .models.yolo import YOLOv8

def detect_growth(image):
    # Initialize the YOLO model
    model = YOLOv8()
    
    # Get predictions
    detections = model.predict(image)
    
    # Post-process the detections if necessary
    processed_detections = post_process_detections(detections)
    
    return processed_detections

def post_process_detections(detections):
    # Implement any required post-processing of the detections
    return detections
