from typing import Optional, Dict, Any
from .input_validation import is_image_file
from .analysis import analyze_and_extract_mic
from .utils import load_image
from .preprocessing import preprocess_images
from .detection import detect_growth
import os

def process_image_file(image_path: str, format_type: str, plate_design: Dict[str, Any], output_directory: str, show: bool = False) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Process a single image file and analyze MIC results.

    :param image_path: Path to the image file to be processed.
    :param format_type: The format in which the MIC results should be saved (e.g., 'csv' or 'json').
    :param plate_design: A dictionary containing the plate design information.
    :param output_directory: The directory where the results should be saved.
    :return: A dictionary of MIC values if the image is processed successfully, or None if the image is invalid.
    """
    if not is_image_file(image_path, plate_design):
        print(f"Skipping non-image file or invalid plate design: {image_path}")
        return

    # Check if the image file name contains '-filtered'
    if '-filtered' in os.path.basename(image_path):
        print(f"Skipping preprocessing for already filtered image: {image_path}")
        processed_image = load_image(image_path)
    else:
        image = load_image(image_path)
        processed_image = preprocess_images(image, image_path=image_path)
    
    detections, inference_time = detect_growth(processed_image)

    # Pass the correct output_directory to analyze_and_extract_mic
    mic_values = analyze_and_extract_mic(image_path, processed_image, detections, plate_design, format_type, output_directory, show)
    return mic_values
