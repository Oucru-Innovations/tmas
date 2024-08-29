import os
from typing import Dict, Any

def is_image_file(file_path: str, plate_design: Dict[str, Any]) -> bool:
    """
    Check if the provided file path is a valid image file based on several criteria.

    This function verifies whether a given file is a valid image file by checking:
    1. The file extension to ensure it is an image format (e.g., .png, .jpg, .jpeg).
    2. The presence of 'raw' or 'filtered' in the filename to ensure it matches expected naming conventions.
    3. The existence of the plate design type extracted from the image name in the provided plate design dictionary.

    :param file_path: The full path to the file being checked.
    :type file_path: str
    :param plate_design: A dictionary representing the plate design loaded from a JSON file, containing design types and related information.
    :type plate_design: Dict[str, Any]
    :return: Returns True if the file is a valid image, contains 'raw' or 'filtered' in the filename, and the plate design type exists in the plate design dictionary. Returns False otherwise.
    :rtype: bool
    """
    # Check for valid image extension
    valid_extensions = ['.png', '.jpg', '.jpeg']
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in valid_extensions:
        # print(f"Invalid file extension for {file_path}")
        return False
    
    # Extract the image name from the file path
    image_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Check for 'raw' or 'filtered' in the filename
    if 'raw' not in image_name.lower() and 'filtered' not in image_name.lower():
        # print(f"Filename does not contain 'raw' or 'filtered': {image_name}")
        return False
    
    # Check if the plate design type exists in the plate design JSON
    split_name = image_name.split('-')
    if len(split_name) > 5:  # Ensure there are enough elements
        plate_design_type = split_name[5]
        if plate_design_type in plate_design:
            return True
        else:
            print(f"Plate design type '{plate_design_type}' not found in the plate design JSON for {image_name}.")
            return False
    else:
        print(f"Error: Image name format is incorrect. Could not extract plate design type from {image_name}.")
        return False
