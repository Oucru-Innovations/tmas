import os
import sys
from tmas.tmas.input_validation import is_image_file  # Updated import
from tmas.tmas.utils import load_plate_design
from tmas.tmas.process_image_file import process_image_file

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python -m scripts.run_tmas <image_path_or_directory> [<format_type>]")
        sys.exit(1)

    path = sys.argv[1]
    format_type = 'csv'  # Default format type is CSV

    # Check if the user provided a format type
    if len(sys.argv) == 3:
        input_format_type = sys.argv[2].lower()
        if input_format_type in ['csv', 'json']:
            format_type = input_format_type
        else:
            print("Invalid format type. Choose 'csv' or 'json'.")
            sys.exit(1)
    
    cwd = os.getcwd()
    print("Current working directory:", cwd)

    plate_design = load_plate_design()  # Load or define plate_design here

    if os.path.isdir(path):
        # Recursively process all images in the directory and subdirectories
        for root, dirs, files in os.walk(path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                print(file_path)
                if is_image_file(file_path, plate_design):
                    print(f"Processing image: {file_path}")
                    process_image_file(file_path, format_type, plate_design)
    elif is_image_file(path, plate_design):
        # Process a single image file
        print(f"Processing image: {path}")
        process_image_file(path, format_type, plate_design)
    else:
        print("Invalid file or directory path. Ensure it exists and contains valid image files.")
        sys.exit(1)

if __name__ == "__main__":
    main()
