from typing import Optional, Dict, Any
from .input_validation import is_image_file  # Updated import
from .analysis import analyze_and_extract_mic
from .utils import load_image
from .preprocessing import preprocess_images
from .detection import detect_growth

def process_image_file(image_path: str, format_type: str, plate_design: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Process a single image file and analyze MIC results.

    :param image_path: Path to the image file to be processed.
    :param format_type: The format in which the MIC results should be saved (e.g., 'csv' or 'json').
    :param plate_design: A dictionary containing the plate design information.
    :return: A dictionary of MIC values if the image is processed successfully, or None if the image is invalid.
    """
    if not is_image_file(image_path, plate_design):
        print(f"Skipping non-image file or invalid plate design: {image_path}")
        return

    image = load_image(image_path)
    processed_image = preprocess_images(image)
    detections, inference_time = detect_growth(processed_image)
    mic_values = analyze_and_extract_mic(image_path, image, detections, plate_design, format_type)
    return mic_values
