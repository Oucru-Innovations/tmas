# tmas/preprocessing.py
import cv2

def preprocess_images(image):
    # Apply Mean Shift Filtering
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
   
    # Apply Pixel Histogram Normalization
    processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Resize with padding for model fitting

    return processed_image
