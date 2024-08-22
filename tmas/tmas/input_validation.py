import os
from typing import Dict, Any

def is_image_file(file_path: str, plate_design: Dict[str, Any]) -> bool:
    """
    Check if the provided file path is a valid image file based on its extension, if it contains 'raw' in the filename,
    and if the plate design type extracted from the image name exists in the plate design JSON.

    :param file_path: The file path to check.
    :param plate_design: The plate design dictionary loaded from JSON.
    :return: True if the file is a valid image, contains 'raw' in the filename, and the plate design type exists, False otherwise.
    """
    # Check for valid image extension
    valid_extensions = ['.png', '.jpg', '.jpeg']
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in valid_extensions:
        return False
    
    # Extract the image name from the file path
    image_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Check if the filename contains 'raw'
    if 'raw' not in image_name.lower():
        return False
    
    # Check if the plate design type exists in the plate design JSON
    try:
        plate_design_type = image_name.split('-')[5]
        if plate_design_type in plate_design:
            return True
        else:
            print(f"Plate design type '{plate_design_type}' not found in the plate design JSON.")
            return False
    except IndexError:
        print("Error: Image name format is incorrect. Could not extract plate design type.")
        return False
