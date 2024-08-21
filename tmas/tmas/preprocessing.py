# tmas/preprocessing.py
import cv2
import numpy as np
import cv2
from PIL import Image, ImageDraw
import numpy as np
from scipy import stats


def convert_image_to_colour(image):

    """
    Convert a grayscale image to a color image by replicating the grayscale values across the RGB channels.

    This function takes a single-channel grayscale image and converts it into a 3-channel color image 
    by assigning the grayscale values to each of the Red, Green, and Blue channels.

    :param image: The input grayscale image.
    :type image: numpy.ndarray

    :return: The converted color image with the same height and width as the input image, 
             but with 3 color channels (RGB).
    :rtype: numpy.ndarray
    """

    # Create an empty 3D array with the same height and width as the original image,
    new_image = np.zeros(image.shape + (3,))

    # Assign the grayscale image to each of the 3 color channels (Red, Green, and Blue).
    for i in [0, 1, 2]:
        new_image[:, :, i] = image

    return new_image

# Function to convert color image to grey scale image
def convert_image_to_grey(image):

    """
    Convert a color image to a grayscale image.

    This function takes a color image with three channels (typically in BGR format) and converts 
    it into a single-channel grayscale image. 

    :param image: The input color image, expected to be in BGR format.
    :type image: numpy.ndarray

    :return: The converted grayscale image.
    :rtype: numpy.ndarray
    """

    new_image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return new_image

# Function to apply a mean shift filter
def mean_shift_filter(image, spatial_radius=10, colour_radius=10):

    """
    Apply a mean shift filter to an image

    This function applies a mean shift filter to the input image. If the input image is not already a 3-channel color image, it will first be converted 
    to color. The mean shift filter is then applied using the specified spatial and color radii.

    :param image: The input image, which can be either a grayscale or color image.
    :type image: numpy.ndarray
    :param spatial_radius: The spatial radius of the mean shift filter. Default is 10.
    :type spatial_radius: int, optional
    :param colour_radius: The color radius of the mean shift filter. Default is 10.
    :type colour_radius: int, optional

    :return: The filtered image after applying the mean shift filter.
    :rtype: numpy.ndarray
    """

    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] != 3):
        image = convert_image_to_colour(image)

    # Apply the mean shift filter
    image = cv2.pyrMeanShiftFiltering(image, spatial_radius, colour_radius)

    return image 

# Function to apply a Contrast Limited Adaptive Histogram Equalization filter.
def equalise_histograms_locally(image, well_dimensions=(8,12)):

    """
    Perform local histogram equalization on an image using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    This function applies local histogram equalization to an image to enhance contrast. If the input image is a color image 
    (3 channels), it will first be converted to grayscale.

    :param image: The input image, which can be either grayscale or color.
    :type image: numpy.ndarray
    :param well_dimensions: The dimensions of the grid used for local histogram equalization. Default is (8, 12).
    :type well_dimensions: tuple[int, int], optional

    :return: The image after applying local histogram equalization.
    :rtype: numpy.ndarray
    """

    # Check if the image is color (3 channels)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = convert_image_to_grey(image)

    # Apply the CLAHE filter for local histogram equalization
    clahe = cv2.createCLAHE(clipLimit= 2.0, tileGridSize=well_dimensions)
    equalised_image = clahe.apply(image)

    return equalised_image


# Function to apply stretch histogram to improve contrast
def stretch_histogram(image):

    """
    Stretch the histogram of an image by adjusting its pixel intensity values.

    This function stretches the histogram of an image by calculating the mode, subtracting it from the 
    image, and then scaling the pixel values based on lower and upper percentiles. 

    :param image: The input image, typically in grayscale, whose histogram will be stretched.
    :type image: numpy.ndarray

    :return: The image with a stretched histogram, with pixel values adjusted to enhance contrast.
    :rtype: numpy.ndarray

    """

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


def preprocess_images(image):

    """
    Preprocess an image by applying a series of filtering and histogram adjustment techniques.

    This function preprocesses an input image by sequentially applying multiple preprocessing steps:

    :param image: The input image, which can be either grayscale or color.
    :type image: numpy.ndarray

    :return: The image after applying preprocessing steps
    :rtype: numpy.ndarray
    """
    
    processed_image = mean_shift_filter(image)

    processed_image = equalise_histograms_locally(processed_image)

    processed_image = stretch_histogram(processed_image)

    return processed_image
