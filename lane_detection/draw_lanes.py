from .helper_functions import *
import cv2
import os
from moviepy.editor import VideoFileClip


def image_processing_pipeline(I_BGR, preview=False, save=False):
    """
    A Pipeline for lane detection of a single image. Results would be saved.

    Input:
    I_BGR: Original BGR image array object

    Output:
    result_Img: BGR image array object with Hough Lines and Vertices drawn onto it.
    """
    output_dir = "./"
    I_Gray = cv2.cvtColor(I_BGR, cv2.COLOR_RGB2GRAY)
    lines, vertices = line_vertices(I_BGR)
    Edge = cv2.Canny(I_Gray, canny_trs_low, canny_trs_high,
                     apertureSize=gaussian_apertureSize, L2gradient=True)
    Edge = region_of_interest(Edge, vertices)
    draw_lines(I_BGR, lines, [34, 126, 230], 3)
    hough_lines_list = hough_lines(Edge, rho, theta, threshold, min_line_length, max_line_gap)
    lane_lines_list = hough2lane_lines(hough_lines_list, I_BGR)
    line_img = np.zeros((I_BGR.shape[0], I_BGR.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lane_lines_list)
    result_Img = weighted_img(line_img, I_BGR)
    if preview:
        image_show(I_BGR)
        image_show(Edge)
        image_show(result_Img)
    if save:
        cv2.imwrite(output_dir+"result.jpg",result_Img)
    return result_Img


def video_processing_pipeline(video_path):
    """
    A video processing pipeline which processes each frame of the video
    with the image processing pipeline defined.
    :param video_path: the path where the video locates
    :return: None
    """
    output_dir = "./"
    Clip=VideoFileClip(video_path)
    result=Clip.fl_image(image_processing_pipeline)
    result.preview(fps=12)
    result.write_videofile(output_dir+"result.mp4", audio=False)


if __name__ == "__main__":
    for f_name in os.listdir("test_images/"):
        I_BGR = cv2.imread("./output/" + f_name)
        image_processing_pipeline(f_name)