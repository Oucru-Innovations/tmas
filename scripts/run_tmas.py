import os
import sys
import argparse
from tmas.src.input_validation import is_image_file
from tmas.src.utils import load_plate_design
from tmas.src.process_image_file import process_image_file

def main():
    parser = argparse.ArgumentParser(description="Run TMAS for image processing, growth detection, and MIC analysis.")
    parser.add_argument("path", type=str, help="Path to the image or directory containing images.")
    parser.add_argument("format_type", type=str, choices=['csv', 'json'], help="Format type for saving results: 'csv' or 'json'.", nargs='?', default='csv')
    parser.add_argument("-visualize", "--visualize", action="store_true", help="Display images with growth detection results.")

    args = parser.parse_args()

    path = args.path
    format_type = args.format_type
    show_images = args.visualize
    
    cwd = os.getcwd()
    # print("Current working directory:", cwd)

    plate_design = load_plate_design()

    if os.path.isdir(path):
        # Track if any filtered images are processed
        filtered_images_exist = False

        # Recursively process all images in the directory and subdirectories
        for root, dirs, files in os.walk(path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                # print(f"Found file: {file_path}")

                # Dynamically set the output directory based on the parent directory of the file
                # Ensure it is always the 'output' subdirectory of the root path
                if 'output' in root.split(os.sep):
                    output_directory = root
                else:
                    output_directory = os.path.join(root, 'output')

                if is_image_file(file_path, plate_design):
                    if '-filtered' in file_name:
                        # Immediately process filtered images
                        filtered_images_exist = True
                        print(f"Processing filtered image: {file_path}")
                        process_image_file(file_path, format_type, plate_design, output_directory, show_images)
        
        if not filtered_images_exist:
            # Process raw images if no filtered images are found
            for root, dirs, files in os.walk(path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    
                    # Ensure it is always the 'output' subdirectory of the root path
                    if 'output' in root.split(os.sep):
                        output_directory = root
                    else:
                        output_directory = os.path.join(root, 'output')

                    if is_image_file(file_path, plate_design) and '-raw' in file_name:
                        print(f"Processing raw image: {file_path}")
                        process_image_file(file_path, format_type, plate_design, output_directory, show_images)
    elif is_image_file(path, plate_design):
        # Dynamically set the output directory based on the parent directory of the image
        if 'output' in os.path.dirname(path).split(os.sep):
            output_directory = os.path.dirname(path)
        else:
            output_directory = os.path.join(os.path.dirname(path), 'output')
        
        # Process a single image file
        print(f"Processing image: {path}")
        process_image_file(path, format_type, plate_design, output_directory, show_images)
    else:
        print("Invalid file or directory path. Ensure it exists and contains valid image files.")
        sys.exit(1)

if __name__ == "__main__":
    main()
