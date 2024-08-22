# scripts/run_tmas.py
from tmas.tmas import preprocess_images, detect_growth, analyze_and_extract_mic
from tmas.tmas.utils import load_image, load_plate_design

# If run from command line, run this: python -m src.scripts.run_tmas

def main():
    # Load image
    image_path = "data/sample_data/01-DR0053-DR0053-1-14-UKMYC6-raw.png"
    image = load_image(image_path)
    
    # Preprocess image
    processed_image = preprocess_images(image)
    
    # Detect growth
    detections, inference_time = detect_growth(processed_image) # return detections growth_matrix and inference_time
    
    # Analyze and extract MIC
    plate_design = load_plate_design()
    mic_values = analyze_and_extract_mic(image_path, image, detections, plate_design)
    
    # Output MIC values
    # print(mic_values)

if __name__ == "__main__":
    main()
