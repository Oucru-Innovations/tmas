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
from typing import Tuple, List, Union, Any, Optional

# Function to resize and pad image to 512x512 with white padding
def resize_with_padding(image: np.ndarray, 
                        target_size: Tuple[int, int] = (512, 512), 
                        padding_color: Tuple[int, int, int] = (255, 255, 255)
                       ) -> Tuple[Image.Image, Tuple[int, int], float]:
    """
    Resize an image while maintaining its aspect ratio, 

    This function resizes the input image so that it fits within the specified `target_size` 
    while preserving the original aspect ratio. It then pads the resized image with a 
    specified `padding_color` to match the `target_size`.

    :param image: The input image as a NumPy array.
    :type image: numpy.ndarray
    :param target_size: The target size for the output image. Defaults to (512, 512).
    :type target_size: tuple[int, int]
    :param padding_color: The color used for padding, specified as an RGB tuple. Defaults to white (255, 255, 255).
    :type padding_color: tuple[int, int, int]

    :return:
        - new_image (PIL.Image.Image): The resized and padded image.
        - padding (tuple[int, int]): The padding applied to the left/right and top/bottom.
        - ratio (float): The scaling ratio used to resize the image.
    :rtype: PIL.Image.Image, tuple[int, int], float

    This function also displays the padded image for debugging purposes.
    """

    print("Resizing and padding the image...")
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(image)
    original_size = image.size
    ratio = float(target_size[0]) / max(original_size)
    new_size = tuple([int(x * ratio) for x in original_size])
    image = image.resize(new_size, Image.Resampling.LANCZOS)
    new_image = Image.new("RGB", target_size, padding_color)
    new_image.paste(image, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))
    padding = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
    
    # Display the padded image for debugging
    # plt.imshow(new_image)
    # plt.title("Padded Image")
    # plt.show()
    
    return new_image, padding, ratio

# def identify_wells(image, hough_param1=10, hough_param2=25, radius_tolerance=0.005, verbose=False):
#     print("Identifying wells...")
#     well_dimensions = (8, 12)
#     well_center = np.zeros((well_dimensions[0], well_dimensions[1], 2), dtype=int)
#     well_radii = np.zeros(well_dimensions, dtype=float)
    
#     image_dimensions = image.shape
#     estimate_well_y = float(image_dimensions[0]) / well_dimensions[0]
#     estimate_well_x = float(image_dimensions[1]) / well_dimensions[1]

#     estimated_radius = (estimate_well_x + estimate_well_y) / 4.
#     max_radius_multiplier = 10.0
#     radius_multiplier = 1. + radius_tolerance
    
#     # Check if the image is already grayscale
#     if len(image.shape) == 2:
#         grey_image = image
#     elif len(image.shape) == 3 and image.shape[2] == 3:
#         grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         raise ValueError("The image must be either a single-channel grayscale or a 3-channel RGB image.")
    
#     plt.imshow(grey_image, cmap='gray')
#     plt.title("Grayscale Image for Well Detection")
#     plt.show()

#     while radius_multiplier <= max_radius_multiplier:
#         circles = cv2.HoughCircles(grey_image, cv2.HOUGH_GRADIENT, 1, 50, 
#                                    param1=hough_param1, param2=hough_param2,
#                                    minRadius=int(estimated_radius / radius_multiplier),
#                                    maxRadius=int(estimated_radius * radius_multiplier))

#         if circles is None or len(circles[0]) == 0:
#             radius_multiplier += radius_tolerance
#             print(f"Radius multiplier increased to {radius_multiplier:.3f}")
#         else:
#             number_of_circles = len(circles[0])
#             print(f"Detected {number_of_circles} circles")
#             if number_of_circles >= well_dimensions[0] * well_dimensions[1]:
#                 for circle in circles[0]:
#                     cx, cy, radius = int(circle[0]), int(circle[1]), circle[2]
#                     col = int(cx / estimate_well_x)
#                     row = int(cy / estimate_well_y)
#                     if row < well_dimensions[0] and col < well_dimensions[1]:
#                         well_center[row, col] = (cx, cy)
#                         well_radii[row, col] = radius
#                 if np.count_nonzero(well_radii) == number_of_wells:
#                     return well_center, well_radii
#             radius_multiplier += radius_tolerance
#             print(f"Radius multiplier increased to {radius_multiplier:.3f}")

#     print("Failed to detect the correct number of wells. Returning False.")
#     return False

# Function to identify wells in the plate -
def identify_wells(image: np.ndarray, 
                   hough_param1: int = 20, 
                   hough_param2: int = 25, 
                   radius_tolerance: float = 0.015, 
                   verbose: bool = False
                  ) -> Union[Tuple[np.ndarray, np.ndarray], bool]:
    """
    Identify the wells on a microtitre plate image.

    This function detects the circular wells on a microtitre plate image, computes their centers
    and radii, and returns this information if the correct number of wells is identified. The
    function assumes a standard 8x12 well layout.

    :param image: The input image of the microtitre plate.
    :type image: numpy.ndarray
    :param hough_param1: The first parameter for the Canny edge detector used in the Hough Circle Transform.
                         Default is 20.
    :type hough_param1: int, optional
    :param hough_param2: The second parameter for the Hough Circle Transform, which controls the minimum
                         number of edge points required to consider a circle valid. Default is 25.
    :type hough_param2: int, optional
    :param radius_tolerance: The tolerance for adjusting the radius during circle detection. Default is 0.015.
    :type radius_tolerance: float, optional
    :param verbose: If True, additional debugging information and images are displayed. Default is False.
    :type verbose: bool, optional

    :return:
        - well_center (numpy.ndarray): A 2D array with the (x, y) coordinates of the
                                       centers of the detected wells.
        - well_radii (numpy.ndarray): A 2D array with the radii of the detected wells.
      If the correct number of wells is not detected, the function returns False.
    :rtype: numpy.ndarray, numpy.ndarray or bool

    :raises ValueError: If the image is not in a valid format (grayscale or RGB) or if the image is empty
                        after processing.
    """

    well_dimensions = (8, 12)
    well_index = np.zeros(well_dimensions, dtype=int)
    well_radii = np.zeros(well_dimensions, dtype=float)
    well_center = np.zeros((well_dimensions[0], well_dimensions[1], 2), dtype=int)
    well_top_left = np.zeros((well_dimensions[0], well_dimensions[1], 2), dtype=int)
    well_bottom_right = np.zeros((well_dimensions[0], well_dimensions[1], 2), dtype=int)

    number_of_wells = well_dimensions[0] * well_dimensions[1]
    image_dimensions = image.shape  # Assumes image is already in grayscale or RGB
    estimate_well_y = float(image_dimensions[0]) / well_dimensions[0]
    estimate_well_x = float(image_dimensions[1]) / well_dimensions[1]

    # Check if the image is already grayscale
    if len(image.shape) == 2:
        grey_image = image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("The image must be either a single-channel grayscale or a 3-channel RGB image.")

    # Ensure the image is in 8-bit format
    grey_image = cv2.convertScaleAbs(grey_image)

    # Check if the image is empty
    if grey_image is None or grey_image.size == 0:
        raise ValueError("The image is empty after processing.")
    
    # plt.imshow(grey_image, cmap='gray')
    # plt.title("Grayscale Image Used for Well Detection")
    # plt.show()

    if estimate_well_x > estimate_well_y:
        if estimate_well_x > 1.05 * estimate_well_y:
            return False
    else:
        if estimate_well_y > 1.05 * estimate_well_x:
            return False

    estimated_radius = (estimate_well_x + estimate_well_y) / 4.
    radius_multiplier = 1. + radius_tolerance

    while True:
        circles = None
        while circles is None:
            circles = cv2.HoughCircles(grey_image, cv2.HOUGH_GRADIENT, 1, 50, 
                                       param1=hough_param1, param2=hough_param2,
                                       minRadius=int(estimated_radius / radius_multiplier),
                                       maxRadius=int(estimated_radius * radius_multiplier))
            radius_multiplier += radius_tolerance

        if circles is not None and len(circles) > 0:
            number_of_circles = len(circles[0])
            if number_of_circles >= number_of_wells:
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
                if top_left[0] < ic[0] < bottom_right[0]:
                    if top_left[1] < ic[1] < bottom_right[1]:
                        number_of_circles_in_well += 1
                        circle = ic

            if number_of_circles_in_well == 1:
                well_centre = (circle[0], circle[1])
                well_radius = circle[2]
                well_extent = 1.2 * well_radius

                well_index[iy, ix] = well_counter
                well_center[iy, ix] = well_centre
                well_radii[iy, ix] = well_radius
                well_top_left[iy, ix] = (max(0, int(well_centre[0] - well_extent)),
                                         max(0, int(well_centre[1] - well_extent)))
                well_bottom_right[iy, ix] = (min(image_dimensions[1], int(well_centre[0] + well_extent)),
                                             min(image_dimensions[0], int(well_centre[1] + well_extent)))
                well_counter += 1
            else:
                one_circle_per_well = False

    if well_counter == number_of_wells and one_circle_per_well:
        return well_center, well_radii
    else:
        return False

def map_predictions_to_plate_design(image: np.ndarray, 
                                    predictions: np.ndarray, 
                                    padding: Tuple[int, int], 
                                    ratio: float, 
                                    image_size: int = 512
                                   ) -> List[List[str]]:
    """
    Map object detection predictions to the corresponding wells in a microtitre plate design.

    This function takes an image of a plate, along with bounding box predictions 
    from an object detection model, and maps these predictions to the appropriate wells 
    on the plate. The function returns a growth matrix indicating which wells show signs of growth.

    :param image: The input image of the plate.
    :type image: numpy.ndarray
    :param predictions: A list of bounding box predictions.
    :type predictions: list[list[float]]
    :param padding: The padding applied to the image during preprocessing
    :type padding: tuple[int, int]
    :param ratio: The scaling ratio applied to the image during resizing.
    :type ratio: float
    :param image_size: The target size of the image after resizing. Default is 512x512 pixels.
    :type image_size: int, optional

    :return: A growth matrix indicating the growth status for each well on the plate.
             Each entry in the matrix is either "-none-" (no growth) or "growth".
    :rtype: list[list[str]]

    :raises ValueError: If the function fails to identify wells in the image.

    """

    print("Mapping predictions to the plate design...")
    c_result = identify_wells(image, hough_param1=20, hough_param2=25, radius_tolerance=0.015, verbose=False)
    
    if c_result is False:
        raise ValueError("Failed to identify wells in the image. Please check the input image or adjust the parameters.")
    
    well_center, well_radii = c_result
    growth_matrix = [["-none-"] * 12 for _ in range(8)]

    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # ax.imshow(image)

    for row in range(8):
        for col in range(12):
            well_x, well_y = well_center[row, col]
            # ax.plot(well_x, well_y, 'bo')  # Plot well center points

    for box in predictions:
        x1, y1, x2, y2 = box[:4]
        x1 = (x1 - padding[0]) / ratio
        y1 = (y1 - padding[1]) / ratio
        x2 = (x2 - padding[0]) / ratio
        y2 = (y2 - padding[1]) / ratio
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        for row in range(8):
            for col in range(12):
                well_x, well_y = well_center[row, col]
                if x1 <= well_x <= x2 and y1 <= well_y <= y2:
                    growth_matrix[row][col] = "growth"

    # plt.axis('off')
    # plt.show()

    return growth_matrix

def post_process_detections(image: np.ndarray, 
                            predictions: np.ndarray, 
                            padding: Tuple[int, int], 
                            ratio: float
                           ) -> List[List[str]]:
    """
    Post-process the detection predictions and map them to the plate design.

    This function takes the bounding box predictions from the object detection model and maps them
    to the corresponding wells on the microtitre plate, resulting in a growth matrix that indicates
    which wells exhibit growth.

    :param image: The input image of the microtitre plate.
    :type image: numpy.ndarray
    :param predictions: A list of bounding box predictions.
    :type predictions: list[list[float]]
    :param padding: The padding applied to the image during preprocessing
    :type padding: tuple[int, int]
    :param ratio: The scaling ratio applied to the image during resizing.
    :type ratio: float

    :return: A growth matrix indicating the growth status for each well on the plate.
             Each entry in the matrix is either "-none-" (no growth) or "growth".
    :rtype: list[list[str]]

    """
    growth_matrix = map_predictions_to_plate_design(image, predictions, padding, ratio)
    return growth_matrix

def detect_growth(image: np.ndarray) -> Tuple[List[List[str]], float]:

    """
    Detect bacterial growth on a plate image using object detection model.

    This function resizes the input image with padding, converts it to grayscale, and then uses 
    a model to detect bacterial growth. The detected bounding boxes are post-processed 
    to map the predictions to the corresponding wells on the plate, resulting in a growth matrix.

    :param image: The input image of the microtitre plate.
    :type image: numpy.ndarray

    :return:
        - growth_matrix (list[list[str]]): A matrix indicating growth status in each well.
        - inference_time (float): The time taken to perform the inference in milliseconds.
    :rtype: list[list[str]], float

    """
    
    print("Detecting growth...")
    model = YOLOv8()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.model.to(device)

    padded_img, padding, ratio = resize_with_padding(image)
    
    padded_img_np = np.array(padded_img)
    
    if len(padded_img_np.shape) == 3 and padded_img_np.shape[2] == 3:
        padded_img_np = cv2.cvtColor(padded_img_np, cv2.COLOR_RGB2GRAY)

    transform = transforms.ToTensor()
    img_tensor = transform(padded_img).to(device).unsqueeze(0)

    start_time = time.time()
    results = model.predict(img_tensor)
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
    scores = results[0].boxes.conf.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()

    predictions = np.hstack((boxes, scores[:, np.newaxis], labels[:, np.newaxis]))

    growth_matrix = post_process_detections(image, predictions, padding, ratio)

    return growth_matrix, inference_time