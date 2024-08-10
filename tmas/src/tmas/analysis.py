# tmas/analysis.py
def analyze_and_extract_mic(detections, plate_design):
    # Implement MIC extraction logic based on detections and plate design
    mic_values = {}
    
    for drug, detection in detections.items():
        mic = extract_mic_from_detection(detection, plate_design[drug])
        mic_values[drug] = mic
        
    return mic_values

def extract_mic_from_detection(detection, drug_concentration):
    # Determine MIC based on growth detection and drug concentration
    # Example logic to find MIC
    for conc in drug_concentration:
        if not detection[conc]:  # No growth detected
            return conc
    return None
