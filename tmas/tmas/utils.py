# tmas/utils.py
import cv2
import json
import os

def load_plate_design():
    config_path = os.path.join(os.path.dirname(__file__), '../../config/plate-design.json')
    with open(config_path, 'r') as file:
        plate_design = json.load(file)
    return plate_design

def load_image(image_path):
    image = cv2.imread(image_path)
    return image
