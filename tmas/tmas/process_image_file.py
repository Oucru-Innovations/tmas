from .input_validation import is_image_file  # Updated import
from .analysis import analyze_and_extract_mic
from .utils import load_image
from .preprocessing import preprocess_images
from .detection import detect_growth
def process_image_file(image_path, format_type, plate_design):
    """
    Process a single image file and analyze MIC results.
    """
    if not is_image_file(image_path, plate_design):
        print(f"Skipping non-image file or invalid plate design: {image_path}")
        return

    image = load_image(image_path)
    processed_image = preprocess_images(image)
    detections, inference_time = detect_growth(processed_image)
    mic_values = analyze_and_extract_mic(image_path, image, detections, plate_design, format_type)
    return mic_values