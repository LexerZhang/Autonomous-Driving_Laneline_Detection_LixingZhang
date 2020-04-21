from moviepy.editor import VideoFileClip
from cam_image_process.image_functions.image_process_pipeline import *
from cam_image_process.image_functions.helper import *


def image_lane_detector(I_BGR, preview=False, save=False):
    """
    A Pipeline for lane detection of a single image. Results would be saved.
    :param I_BGR: Original BGR image array object
    :return lane_line_list: BGR image array object with Hough Lines and Vertices drawn onto it.
    """
    I_Gray = cv2.cvtColor(I_BGR, cv2.COLOR_RGB2GRAY)
    lines, vertices = line_vertices(I_BGR)
    Edge = cv2.Canny(I_Gray, canny_trs_low, canny_trs_high,
                     apertureSize=gaussian_apertureSize, L2gradient=True)
    Edge = region_of_interest(Edge, vertices)
    helper.draw_lines(I_BGR, lines, [34, 126, 230], 3)
    hough_lines_list = hough_lines(Edge, rho, theta, threshold, min_line_length, max_line_gap)
    lane_lines_list = hough2lane_lines(hough_lines_list, I_BGR)
    if preview:
        marked_image_output(I_BGR, lane_lines_list, save=save)
    return lane_lines_list


def marked_image_output(img, lane_lines_list=None, preview=True, output_dir='./', save=True):
    """
    Preview and save the image frame with features calculated.
    :param img: The original RGB image
    :param lane_lines_list:
    :param output_dir:
    :return:
    """
    if lane_lines_list is None:
        lane_lines_list = image_lane_detector(img)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    helper.draw_lines(line_img, lane_lines_list)
    result_img = helper.weighted_img(line_img, img)
    if preview:
        helper.image_show(result_img)
    if save:
        cv2.imwrite(output_dir + "result.jpg", result_img)
    return result_img


def marked_video_output(video_path, output_dir = "./"):
    """
    A video processing pipeline which processes each frame of the video
    with the image processing pipeline defined.
    :param video_path: the path where the video locates
    :return: None
    """
    clip = VideoFileClip(video_path)
    result = clip.fl_image(lambda a: marked_image_output(a, image_lane_detector(a), preview=False, save=False))
    result.preview(fps=12)
    result.write_videofile(output_dir + "result.mp4", audio=False)


if __name__ == "__main__":
    img1 = cv2.imread('test_images/solidWhiteCurve.jpg')
    f_l = FeatureCollector(img1)
    f_l.image_show()
