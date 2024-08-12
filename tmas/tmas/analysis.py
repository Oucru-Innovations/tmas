# tmas/analysis.py
import matplotlib.patches as patches
import cv2
import matplotlib.pyplot as plt
import os
from .detection import identify_wells

def build_drug_info_dict(plate_info):
    drug_info = {}
    drug_matrix = plate_info["drug_matrix"]
    dilution_matrix = plate_info["dilution_matrix"]
    conc_matrix = plate_info["conc_matrix"]

    for row in range(8):
        for col in range(12):
            drug = drug_matrix[row][col]
            if drug not in drug_info:
                drug_info[drug] = {"dilutions": [], "concentrations": []}
            drug_info[drug]["dilutions"].append(dilution_matrix[row][col])
            drug_info[drug]["concentrations"].append(conc_matrix[row][col])

    # Sort the dilutions and concentrations for each drug
    for drug in drug_info:
        combined = list(zip(drug_info[drug]["dilutions"], drug_info[drug]["concentrations"]))
        combined.sort()  # Sort by dilution
        drug_info[drug]["dilutions"], drug_info[drug]["concentrations"] = zip(*combined)

    return drug_info

def visualize_growth_matrix(image_name, img, growth_matrix, drug_info, drug_results, plate_info):
    # Convert the image to RGB if it's not already
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # plt.imshow(img)
    # plt.title("Image Alone")
    # plt.show()

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)

    # Identify wells on the plate
    well_center, well_radii = identify_wells(img, hough_param1=20, hough_param2=25, radius_tolerance=0.015, verbose=False)

    # Extract drug and concentration matrices from plate information
    drug_matrix = plate_info["drug_matrix"]
    conc_matrix = plate_info["conc_matrix"]

    # Iterate over each well to visualize growth and drug information
    for row in range(8):
        for col in range(12):
            drug = drug_matrix[row][col]
            conc = conc_matrix[row][col]
            growth = growth_matrix[row][col]

            well_x, well_y = well_center[row, col]
            if growth == 'growth':
                # Draw a green circle to indicate growth
                circle = patches.Circle((well_x, well_y), well_radii[row, col] * 0.9, linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(circle)

            # Add drug name and concentration text annotations
            ax.text(well_x, well_y - well_radii[row, col] * 0.3, drug, ha='center', va='center', fontsize=10, color='black', alpha=0.8)
            ax.text(well_x, well_y + well_radii[row, col] * 0.3, conc, ha='center', va='center', fontsize=10, color='black', alpha=0.8)

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Growth Detection Matrix for {image_name}")
    plt.axis('off')

    # Save the figure to a file with the image name
    plt.savefig(f"{image_name}_growth_matrix.png", bbox_inches='tight')
    plt.show()

    # Print MIC results
    print(f"MIC Results for {image_name}:")
    for drug, result in drug_results.items():
        print(f"{drug}: {result['growth_array']}, MIC: {result['MIC']}")

def analyze_growth_matrix(image_name, image, growth_matrix, plate_design, plate_design_type): # image_name, image, detections, plate_design, plate_design_type
    print(f"Current plate design: {plate_design_type}")

    drug_info = build_drug_info_dict(plate_design)

    if growth_matrix[7][10] != 'growth' and growth_matrix[7][11] != 'growth':
        print("Positive Control Wells - Invalid")
        return

    drug_growth = {drug: ['-none-'] * len(info["dilutions"]) for drug, info in drug_info.items() if drug != "POS"}

    for row in range(8):
        for col in range(12):
            drug = plate_design["drug_matrix"][row][col]
            if drug == "POS":
                continue  # Skip POS wells
            dilution = plate_design["dilution_matrix"][row][col]
            index = drug_info[drug]["dilutions"].index(dilution)
            if growth_matrix[row][col] == 'growth':
                drug_growth[drug][index] = 'growth'

    num_drugs = len(drug_growth)
    print(f"Number of drugs (excluding POS): {num_drugs}")

    drug_results = {}
    for drug, growth_array in drug_growth.items():
        all_growth = all(x == 'growth' for x in growth_array)
        all_none = all(x == '-none-' for x in growth_array)

        if all_growth:
            mic = f">= {drug_info[drug]['concentrations'][-1]}"  # MIC is greater than or equal to the highest concentration
        elif all_none:
            mic = f"<= {drug_info[drug]['concentrations'][0]}"  # MIC is less than or equal to the lowest concentration
        else:
            for i in range(len(growth_array)):
                if growth_array[i] == '-none-':
                    mic = drug_info[drug]['concentrations'][i]
                    break
            else:
                mic = "Invalid"

        drug_results[drug] = {"growth_array": growth_array, "MIC": mic}

    visualize_growth_matrix(image_name, image, growth_matrix, drug_info, drug_results, plate_design)
    return drug_results

def extract_image_name_from_path(image_path):
    # Extract the file name from the image path
    return os.path.splitext(os.path.basename(image_path))[0]

def extract_plate_design_type_from_image_name(image_name):
    # Assuming the plate design is always the 6th element in the hyphen-separated image name
    return image_name.split('-')[5]

def analyze_and_extract_mic(image_path, image, detections, plate_design):
    image_name = extract_image_name_from_path(image_path) # 02-1090-2013185209-1-14-UKMYC5-raw
    plate_design_type = extract_plate_design_type_from_image_name(image_name)    
    # Use the plate_design_type string to access the correct dictionary within plate_designs
    drug_results = analyze_growth_matrix(image_name, image, detections, plate_design[plate_design_type], plate_design_type)

    return drug_results