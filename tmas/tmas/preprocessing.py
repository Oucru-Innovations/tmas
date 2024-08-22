import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import stats
from typing import Tuple

# Function to convert a grayscale image to a color image.
def convert_image_to_colour(image: np.ndarray) -> np.ndarray:
    # Create an empty 3D array with the same height and width as the original image,
    new_image = np.zeros(image.shape + (3,))

    # Assign the grayscale image to each of the 3 color channels (Red, Green, and Blue).
    for i in [0, 1, 2]:
        new_image[:, :, i] = image

    return new_image

# Function to convert color image to grey scale image
def convert_image_to_grey(image: np.ndarray) -> np.ndarray:
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return new_image

# Function to apply a mean shift filter
def mean_shift_filter(image: np.ndarray, spatial_radius: int = 10, colour_radius: int = 10) -> np.ndarray:
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] != 3):
        image = convert_image_to_colour(image)

    # Apply the mean shift filter
    image = cv2.pyrMeanShiftFiltering(image, spatial_radius, colour_radius)

    return image  # Return the filtered image

# Function to apply a Contrast Limited Adaptive Histogram Equalization filter.
def equalise_histograms_locally(image: np.ndarray, well_dimensions: Tuple[int, int] = (8, 12)) -> np.ndarray:
    # Check if the image is color (3 channels)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = convert_image_to_grey(image)

    # Apply the CLAHE filter for local histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=well_dimensions)
    equalised_image = clahe.apply(image)

    return equalised_image

# Function to apply stretch histogram to improve contrast
def stretch_histogram(image: np.ndarray) -> np.ndarray:
    # Calculate the mode of the image
    mode = stats.mode(image, axis=None)[0]

    # Subtract the mode from the image
    image = (np.array(image, dtype=np.int16)) - mode

    # Calculate the lower and upper percentiles after subtraction
    lower = np.percentile(image, 5)
    upper = np.percentile(image, 95)

    # Determine scaling factors
    pos_factor = 40. / upper
    neg_factor = -110. / lower

    # Apply scaling factors based on whether pixel values are positive or negative
    image = np.multiply(image, np.where(image > 0, pos_factor, neg_factor))

    # Offset the image to ensure positive pixel values
    image = image + 180.

    return image

# Function to preprocess images with a series of filters
def preprocess_images(image: np.ndarray) -> np.ndarray:
    processed_image = mean_shift_filter(image)
    processed_image = equalise_histograms_locally(processed_image)
    processed_image = stretch_histogram(processed_image)

    return processed_image