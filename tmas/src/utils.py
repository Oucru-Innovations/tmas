import cv2
import json
import os
from typing import Dict, Any
import numpy as np

def load_plate_design() -> Dict[str, Any]:
    """
    Load the plate design configuration from a JSON file.

    :return: A dictionary containing the plate design configuration.
    """
    config_path = os.path.join(os.path.dirname(__file__), '../../config/plate-design.json')
    print("Looking for plate-design.json at:", config_path)

    with open(config_path, 'r') as file:
        plate_design = json.load(file)
        print("Successfully loaded file.")
    return plate_design

def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the specified file path.

    :param image_path: The path to the image file.
    :return: The loaded image as a NumPy array.
    """
    image = cv2.imread(image_path)
    return image
