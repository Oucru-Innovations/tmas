# tmas/analysis.py
import matplotlib.patches as patches
import cv2
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import csv
import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional
from .detection import identify_wells

def build_drug_info_dict(plate_info: Dict[str, List[List[Union[str, float]]]]) -> Dict[str, Dict[str, List[Union[str, float]]]]:
    """
    Build a dictionary containing drug information based on plate data.

    This function processes the `plate_info` dictionary. It returns a
    dictionary where each drug is a key, and the corresponding value is another
    dictionary containing sorted lists of dilutions and concentrations.


    :param plate_info: A dictionary containing the following keys:
        - "drug_matrix":
        - "dilution_matrix": 
        - "conc_matrix": 
    :type plate_info: dict

    :return: A dictionary where each key is a drug name, and the value is another dictionary
             containing:
             - "dilutions"
             - "concentrations"
    :rtype: dict

    :raises KeyError: If any of the expected keys ("drug_matrix", "dilution_matrix", "conc_matrix")
                      are missing from `plate_info`.
    """
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

def visualize_growth_matrix(image_name: str,
                            img: Any,
                            growth_matrix: List[List[str]],
                            drug_info: Dict[str, Dict[str, List[Union[str, float]]]],
                            drug_results: Dict[str, Dict[str, Union[str, List[str]]]],
                            plate_info: Dict[str, List[List[Union[str, float]]]],
                            output_directory: str,
                            show: bool = False) -> None:
    """
    Visualize the growth matrix on a drug susceptibility testing plate image.

    This function overlays the growth detection information on an image of a microtitre plate,
    highlighting wells with detected growth, and annotating the wells with the corresponding drug
    names and concentrations. The visualized image is saved to the specified output directory, 
    and optionally displayed based on the `show` parameter.

    :param image_name: The base name of the image, used for labeling and saving results.
    :type image_name: str
    :param img: The image of the plate. If the image is in grayscale or BGR format, it will be converted to RGB.
    :type img: numpy.ndarray
    :param growth_matrix: A matrix indicating the growth status of each well ('growth' or '-none-').
    :type growth_matrix: list[list[str]]
    :param drug_info: A dictionary containing drug information, including dilutions and concentrations for each drug.
    :type drug_info: dict
    :param drug_results: A dictionary containing the results of the drug susceptibility test, including MIC values for each drug.
    :type drug_results: dict
    :param plate_info: A dictionary containing plate-specific information, including drug and concentration matrices.
    :type plate_info: dict
    :param output_directory: The directory where the visualization image will be saved.
    :type output_directory: str
    :param show: Boolean flag indicating whether to display the image after saving. If True, the image is displayed.
    :type show: bool
    :return: None
    :rtype: None
    """
    
    # Convert image to uint8 if it is not already
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)

    # Check the image format
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Input image format is not supported. Expected grayscale or RGB/BGR image.")

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 6))
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
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust these values to reduce padding

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Growth Detection Matrix for {image_name}")
    plt.axis('off')

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Save the figure to a file directly in the specified output directory
    image_path = os.path.join(output_directory, f"{image_name}-growth-matrix-visualization.png")
    plt.savefig(image_path, bbox_inches='tight')
    
    # Show the image only if the show flag is True
    if show:
        plt.show()

    # Print MIC results
    print(f"MIC Results for {image_name}:")
    for drug, result in drug_results.items():
        print(f"{drug}: {result['growth_array']}, MIC: {result['MIC']}")

def save_mic_results(data: List[Dict[str, Union[str, float]]],
                    format_type: str,
                    filename: str,
                    output_directory: str) -> None:
    """
    Save MIC results in the specified format to the specified directory,
    appending to existing files if they already exist.

    :param data: List of dictionaries containing drug, results, MIC, and image name.
    :param format_type: 'csv' or 'json' for the file format.
    :param filename: Base filename for the output file.
    :param output_directory: The directory where the results should be saved.
    :type output_directory: str
    """
    
    # print(f'Output directory for saving MIC results: {output_directory}')
    
    os.makedirs(output_directory, exist_ok=True)  # Ensure the output directory exists
    full_path = os.path.join(output_directory, filename.replace('-raw', '-mic'))  # Ensure correct output path

    try:
        if format_type == 'csv':
            # Convert data to DataFrame
            df = pd.DataFrame(data)
            df.to_csv(f"{full_path}.csv", index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"Data saved as {full_path}.csv")
        
        elif format_type == 'json':
            # Prepare data for JSON: use filename as key
            json_data = {filename: data}
            # Save data as JSON
            with open(f"{full_path}.json", 'w') as json_file:
                json.dump(json_data, json_file, indent=4)
            print(f"Data saved as {full_path}.json")
    except Exception as e:
        print(f"An error occurred while saving the file: {str(e)}")

def analyze_growth_matrix(image_name: str,
                          image: Any,
                          growth_matrix: List[List[str]],
                          plate_design: Dict[str, List[List[Union[str, float]]]],
                          plate_design_type: str,
                          output_directory: str,
                          show: bool = False) -> Optional[Dict[str, Dict[str, Union[str, List[str]]]]]:
    """
    Analyze the growth matrix and determine the Minimum Inhibitory Concentration (MIC) for each drug.

    This function processes the growth matrix of a microtitre plate, compares it with the provided plate design, 
    and calculates the MIC values for each drug tested. It also visualizes the results by overlaying growth detection
    information on the image of the plate, optionally displaying the visualization.

    :param image_name: The name of the image file used for labeling and saving results.
    :type image_name: str
    :param image: The image data of the plate, typically as a NumPy array.
    :type image: numpy.ndarray
    :param growth_matrix: A matrix indicating the growth status of each well (e.g., 'growth' or '-none-').
    :type growth_matrix: list[list[str]]
    :param plate_design: A dictionary containing the design of the plate, including drug, dilution, and concentration matrices.
    :type plate_design: dict
    :param plate_design_type: A string representing the type of plate design being used.
    :type plate_design_type: str
    :param output_directory: The directory where the visualization image and results should be saved.
    :type output_directory: str
    :param show: A boolean flag indicating whether to display the visualization of the growth matrix.
    :type show: bool

    :return: A dictionary containing the growth results and MIC values for each drug. The dictionary keys are drug names,
             and the values are dictionaries with 'growth_array' and 'MIC' keys.
    :rtype: Optional[dict]

    :raises ValueError: If the image is not in a supported format or if there are issues with the plate design data.
    """
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

    visualize_growth_matrix(image_name, image, growth_matrix, drug_info, drug_results, plate_design, output_directory, show)
    return drug_results

def extract_image_name_from_path(image_path: str) -> str:
    """
    Extract the base name of an image file from a file path and remove specific suffixes.

    This function takes a full file path to an image, extracts the base name of the file
    without its extension, and removes the suffixes '-filtered' or '-raw' from the base name
    if they are present. This is useful for standardizing image file names during processing.

    :param image_path: The full file path to the image file.
    :type image_path: str

    :return: The standardized base name of the image file without the extension and without '-filtered' or '-raw' suffixes.
    :rtype: str
    """
    # Extract the file name from the image path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Remove '-filtered' or '-raw' from the base name if present
    base_name = base_name.replace('-filtered', '').replace('-raw', '')
    
    return base_name


def extract_plate_design_type_from_image_name(image_name: str) -> str:
    """
    Extract the plate design type from an image name.

    Assumes that the plate design type is always the 6th element in a hyphen-separated image name string.

    :param image_name: The image name string, typically in a hyphen-separated format.
    :type image_name: str

    :return: The plate design type extracted from the image name.
    :rtype: str
    """
    # Assuming the plate design is always the 6th element in the hyphen-separated image name
    return image_name.split('-')[5]

def analyze_and_extract_mic(image_path: str,
                            image: Any,
                            detections: List[List[str]],
                            plate_design: Dict[str, Dict[str, List[List[Union[str, float]]]]],
                            format_type: str,
                            output_directory: str,
                            show: bool = False) -> Optional[Dict[str, Dict[str, Union[str, List[str]]]]]:
    """
    Analyze the growth matrix and extract Minimum Inhibitory Concentration (MIC) results.

    This function analyzes the growth detection results from an image of a microtitre plate to determine the 
    Minimum Inhibitory Concentration (MIC) values for various drugs. It processes the image data and detection results 
    based on the provided plate design and saves the results in the specified format.

    :param image_path: The full file path to the image to be processed.
    :type image_path: str
    :param image: The image data as a NumPy array, typically representing a microtitre plate.
    :type image: Any
    :param detections: A list of detection results indicating the growth status for each well on the plate.
    :type detections: List[List[str]]
    :param plate_design: A dictionary containing the plate design information, indexed by plate design type.
    :type plate_design: Dict[str, Dict[str, List[List[Union[str, float]]]]]
    :param format_type: The format in which the MIC results should be saved, either 'csv' or 'json'.
    :type format_type: str
    :param output_directory: The directory where the processed results and visualization should be saved.
    :type output_directory: str
    :param show: A boolean flag indicating whether to display the visualization images. Defaults to False.
    :type show: bool
    :return: A dictionary containing the MIC results and growth information for each drug, or None if the analysis fails.
    :rtype: Optional[Dict[str, Dict[str, Union[str, List[str]]]]]
    """
    image_name = extract_image_name_from_path(image_path)
    plate_design_type = extract_plate_design_type_from_image_name(image_name)
    
    # Use the provided output_directory
    drug_results = analyze_growth_matrix(image_name, image, detections, plate_design[plate_design_type], plate_design_type, output_directory, show)

    # Save MIC results directly to the provided output_directory
    filename = image_name  # Use cleaned image name as the filename
    if format_type == 'csv':
        data_with_format = [
            {"Filename": filename, "Drug": drug, "Results": ', '.join(result["growth_array"]), "MIC": result["MIC"]}
            for drug, result in drug_results.items()
        ]
        save_mic_results(data_with_format, format_type, filename, output_directory)
    else:
        data = [
            {"Drug": drug, "Results": ', '.join(result["growth_array"]), "MIC": result["MIC"]}
            for drug, result in drug_results.items()
        ]
        save_mic_results(data, format_type, filename, output_directory)

    return drug_results
