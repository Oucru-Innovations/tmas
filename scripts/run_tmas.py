# scripts/run_tmas.py
from src.tmas import preprocess_images, detect_growth, analyze_and_extract_mic
from src.tmas.utils import load_image, load_plate_design

# If run from command line, run this: python -m src.scripts.run_tmas

def main():
    # Load image
    image = load_image("data/sample_data/01-DR0013-DR0013-1-14.png")
    
    # Preprocess image
    # processed_image = preprocess_images(image)
    processed_image = image
    
    # Detect growth
    detections = detect_growth(processed_image) # return {boxes, labels, scores, inference_time}
    
    # Analyze and extract MIC
    plate_design = load_plate_design()
    mic_values = analyze_and_extract_mic(detections, plate_design)
    
    # Output MIC values
    print(mic_values)

if __name__ == "__main__":
    main()
