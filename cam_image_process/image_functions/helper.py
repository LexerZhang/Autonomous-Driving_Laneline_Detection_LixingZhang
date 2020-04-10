"""Provide some general image_functions for image processing."""
import cv2
import numpy as np


def image_show(img, name='image'):
    """Show an image until any key is pushed."""
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def draw_lines(img, lines, color=(0, 255, 0), thickness=6):
    """
    TODO:This function draws `lines` with `color` and `thickness`, creating a blank image with lines drawn onto.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def image_normalization(img, abs=True):
    """
    Scale an image to a uint8 array for representation.
    :param img: single or multi-channel image
    :return: the scaled image matrix in format uint8
    """

    if abs:
        img = np.abs(np.int16(img))
    val_max = img.max()
    val_min = img.min()
    return np.uint8((img - val_min) * 255 / (val_max - val_min))
