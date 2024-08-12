# src/tmas/models/yolo.py
import os
import requests
from ultralytics import YOLO

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def download_model():
    url = 'https://drive.google.com/uc?export=download&id=17s_7WR2YgKkhigQEY92o740iwXYjiXr0'
    local_filename = "tmas/tmas/models/best_model_yolo.pt"

    session = requests.Session()
    response = session.get(url, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'confirm': token}
        response = session.get(url, params=params, stream=True)

    os.makedirs(os.path.dirname(local_filename), exist_ok=True)  # Create directory if it doesn't exist
    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(32768):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    print(f"Model downloaded successfully to {local_filename}")

class YOLOv8:
    def __init__(self, weights="tmas/tmas/models/best_model_yolo.pt"):
        if not os.path.exists(weights):
            print(f"Model file {weights} not found. Attempting to download...")
            download_model()
        
        # Load the YOLOv8 model
        self.model = YOLO(weights)

    def predict(self, img):
        return self.model(img)
