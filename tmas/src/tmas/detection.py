# tmas/detection.py
from .models.yolo import YOLOv8
from PIL import Image
import numpy as np
import torch
import time
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to resize and pad image to 512x512 with white padding
def resize_with_padding(image, target_size=(512, 512), padding_color=(255, 255, 255)):
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image) 
    original_size = image.size  
    ratio = float(target_size[0]) / max(original_size)
    new_size = tuple([int(x * ratio) for x in original_size]) 
    image = image.resize(new_size, Image.Resampling.LANCZOS)  
    new_image = Image.new("RGB", target_size, padding_color)  
    new_image.paste(image, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))  
    padding = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
    return new_image, padding, ratio

# Function to identify wells in the plate
def identify_wells(image, hough_param1=20, hough_param2=25, radius_tolerance=0.005, verbose=False):
    well_dimensions = (8, 12)
    well_index = np.zeros(well_dimensions, dtype=int)
    well_radii = np.zeros(well_dimensions, dtype=float)
    well_center = np.zeros((well_dimensions[0], well_dimensions[1], 2), dtype=int)

    number_of_wells = well_dimensions[0] * well_dimensions[1]
    image_dimensions = image.shape  # Assumes image is already in grayscale
    estimate_well_y = float(image_dimensions[0]) / well_dimensions[0]
    estimate_well_x = float(image_dimensions[1]) / well_dimensions[1]

    if estimate_well_x > estimate_well_y:
        if estimate_well_x > 1.05 * estimate_well_y:
            return False
    else:
        if estimate_well_y > 1.05 * estimate_well_x:
            return False

    estimated_radius = (estimate_well_x + estimate_well_y) / 4.
    radius_multiplier = 1. + radius_tolerance
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    while True:
        circles = None
        while circles is None:
            circles = cv2.HoughCircles(grey_image, cv2.HOUGH_GRADIENT, 1, 50, param1=hough_param1, param2=hough_param2,
                                       minRadius=int(estimated_radius / radius_multiplier),
                                       maxRadius=int(estimated_radius * radius_multiplier))
            radius_multiplier += radius_tolerance

        number_of_circles = len(circles[0])

        if number_of_circles >= number_of_wells:
            break
        elif number_of_circles > number_of_wells:
            break
        elif radius_multiplier > 2:
            break
        else:
            radius_multiplier += radius_tolerance

    well_counter = 0
    one_circle_per_well = True

    for ix in range(0, well_dimensions[1]):
        for iy in range(0, well_dimensions[0]):
            top_left = (int(ix * estimate_well_x), int(iy * estimate_well_y))
            bottom_right = (int((ix + 1) * estimate_well_x), int((iy + 1) * estimate_well_y))
            number_of_circles_in_well = 0

            for ic in circles[0, ]:
                if top_left[0] < ic[0] < bottom_right[0] and top_left[1] < ic[1] < bottom_right[1]:
                    number_of_circles_in_well += 1
                    circle = ic

            if number_of_circles_in_well == 1:
                well_centre = (circle[0], circle[1])
                well_radius = circle[2]

                well_index[iy, ix] = well_counter
                well_center[iy, ix] = well_centre
                well_radii[iy, ix] = well_radius
                well_counter += 1
            else:
                one_circle_per_well = False

    if well_counter == number_of_wells and one_circle_per_well:
        return well_center, well_radii
    else:
        return False

# Function to map bounding boxes to plate design wells
def map_predictions_to_plate_design(image, predictions, padding, ratio, image_size=512):
    well_center, well_radii = identify_wells(image, hough_param1=20, hough_param2=25, radius_tolerance=0.015, verbose=False)
    growth_matrix = [["-none-"] * 12 for _ in range(8)]

    if well_center is False:
        print(f"Failed to identify wells for image {image}")
        return growth_matrix

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)

    for row in range(8):
        for col in range(12):
            well_x, well_y = well_center[row, col]
            ax.plot(well_x, well_y, 'bo')  # Plot well center points

    for box in predictions:
        x1, y1, x2, y2 = box[:4]
        x1 = (x1 - padding[0]) / ratio
        y1 = (y1 - padding[1]) / ratio
        x2 = (x2 - padding[0]) / ratio
        y2 = (y2 - padding[1]) / ratio
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        for row in range(8):
            for col in range(12):
                well_x, well_y = well_center[row, col]
                if x1 <= well_x <= x2 and y1 <= well_y <= y2:
                    growth_matrix[row][col] = "growth"

    plt.axis('off')
    plt.show()

    return growth_matrix

# Post-process the detections to map them to the plate design
def post_process_detections(image, predictions, padding, ratio):
    growth_matrix = map_predictions_to_plate_design(image, predictions, padding, ratio)
    return growth_matrix

# Function to detect growth and return the processed growth matrix and inference time
def detect_growth(image): 
    # Initialize the YOLO model
    model = YOLOv8()

    # Check if CUDA is available and use it if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Resize the image to 512x512 with padding
    padded_img, padding, ratio = resize_with_padding(image)
    
    # Convert the image to a tensor
    transform = transforms.ToTensor()
    img_tensor = transform(padded_img).to(device).unsqueeze(0)

    # Measure inference time
    start_time = time.time()
    # Perform detection
    print("checkkkkkk----")
    results = model.predict(img_tensor)  # Use the predict method here
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    # Extract boxes, labels, and scores
    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
    scores = results[0].boxes.conf.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()

    # Combine predictions into a single array
    predictions = np.hstack((boxes, scores[:, np.newaxis], labels[:, np.newaxis]))

    # Post-process the detections to map them to the plate design
    growth_matrix = post_process_detections(image, predictions, padding, ratio)

    return growth_matrix, inference_time
