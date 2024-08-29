from typing import Optional, Dict, Any
from .input_validation import is_image_file
from .analysis import analyze_and_extract_mic
from .utils import load_image
from .preprocessing import preprocess_images
from .detection import detect_growth
import os

def process_image_file(image_path: str, format_type: str, plate_design: Dict[str, Any], output_directory: str, show: bool = False) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Process a single image file to detect bacterial growth and analyze MIC (Minimum Inhibitory Concentration) results.

    This function checks the validity of an image file, applies necessary preprocessing steps, 
    performs growth detection using an object detection model, and then analyzes the results 
    to calculate MIC values for various drugs. The results are saved in the specified format.

    :param image_path: The full path to the image file to be processed.
    :type image_path: str
    :param format_type: The format for saving MIC results, either 'csv' or 'json'.
    :type format_type: str
    :param plate_design: A dictionary containing the design details of the plate, including drug and dilution matrices.
    :type plate_design: Dict[str, Any]
    :param output_directory: The directory path where the output results and visualizations will be saved.
    :type output_directory: str
    :param show: A boolean flag that determines whether to display the visualization images. Defaults to False.
    :type show: bool
    :return: A dictionary containing MIC values if the image is processed successfully, or None if the image is invalid.
    :rtype: Optional[Dict[str, Dict[str, Any]]]
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
