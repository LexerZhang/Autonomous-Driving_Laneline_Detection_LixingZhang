from .helper_functions import *
import cv2
import os
from moviepy.editor import VideoFileClip


def image_processing_pipeline(I_BGR):
    """
    A Pipeline for lane detection of a single image. Results would be saved.

    Input:
    I_BGR: Original BGR image array object

    Output:
    result_Img: BGR image array object with Hough Lines and Vertices drawn onto it.
    """
    output_dir = "./output/"
    I_Gray = cv2.cvtColor(I_BGR, cv2.COLOR_RGB2GRAY)
    lines, vertices = line_vertices(I_BGR)
    image_show(I_BGR)
    Edge = cv2.Canny(I_Gray, canny_trs_low, canny_trs_high,
                     apertureSize=gaussian_apertureSize, L2gradient=True)
    Edge = region_of_interest(Edge, vertices)
    image_show(Edge)
    draw_lines(I_BGR, lines, [34, 126, 230], 3)
    hough_lines_list = hough_lines(Edge, rho, theta, threshold, min_line_length, max_line_gap)
    lane_lines_list = hough2lane_lines(hough_lines_list, I_BGR)
    line_img = np.zeros((I_BGR.shape[0], I_BGR.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lane_lines_list)
    result_Img = weighted_img(line_img, I_BGR)
    image_show(result_Img)
    return result_Img


def video_processing_pipeline():
    pass


if __name__ == "__main__":
    for f_name in os.listdir("test_images/"):
        I_BGR = cv2.imread("./output/" + f_name)
        image_processing_pipeline(f_name)