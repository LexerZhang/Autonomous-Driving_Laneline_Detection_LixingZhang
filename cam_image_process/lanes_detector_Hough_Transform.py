from moviepy.editor import VideoFileClip
from .image_functions.image_process_pipeline import *
from .image_functions.helper import *


def Hough_image_lane_detector(img_container):
    """
    A Pipeline for lane detection of a single image. Results would be saved.
    :param I_BGR: Original BGR image array object
    :return lane_line_list: BGR image array object with Hough Lines and Vertices drawn onto it.
    """
    #img1 = cv2.imread('test_images/solidYellowCurve.jpg') #TODO: Remove later
    #img_container = FeatureCollector(img1) #TODO: Remove later
    vertices, edge_lines = get_vertices(img_container.img)

    img_container.add_layer("region_mask",'mask')
    img_container.layers_dict["region_mask"].geometrical_mask(vertices)

    img_container.add_layer("edge_lines","mask")
    img_container.layers_dict["edge_lines"].straight_lines(edge_lines)

    S_sobel_mag = ImgFeature3(img_container.img)
    S_sobel_mag.channel_selection('S').sobel_convolute("mag").binary_threshold((50,200))
    img_container.add_layer("S_sobel_mag", layer=S_sobel_mag)

    S_sobel_dir = ImgFeature3(img_container.img)
    S_sobel_dir.channel_selection('S').sobel_convolute("dir").binary_threshold((np.pi*2/9, np.pi/3))
    img_container.add_layer("S_sobel_dir", layer=S_sobel_dir)

    img_container.combine("S_sobel_mag","S_sobel_dir","and")
    img_container.combine("S_sobel_mag", "region_mask","and")
    img_container.image_show(True)
    img_container.combine("main","edge_lines", "mix")
    img_container.image_show(True)
    #helper.draw_lines(I_BGR, lines, [34, 126, 230], 3)
    #hough_lines_list = hough_lines(Edge, rho, theta, threshold, min_line_length, max_line_gap)
    #lane_lines_list = hough2lane_lines(hough_lines_list, I_BGR)
    return img_container.img_processed


'''def marked_image_pipeline(img, lane_lines_list=None, preview=True, output_dir='./', save=True):
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
    return result_img'''


'''def marked_video_output(video_path, output_dir = "./"):
    """
    A video processing pipeline which processes each frame of the video
    with the image processing pipeline defined.
    :param video_path: the path where the video locates
    :return: None
    """
    clip = VideoFileClip(video_path)
    result = clip.fl_image(lambda a: marked_image_output(a, image_lane_detector(a), preview=False, save=False))
    result.preview(fps=12)
    result.write_videofile(output_dir + "result.mp4", audio=False)'''


def get_vertices(img_BGR):
    """
    Accepts an RGB image array, return a tuple of (lines, vertices) for region selection.

    Input:
    img_RGB: 3-tunnel image array, with size of Height * Width * Tunnels

    Output:
    lines: cordinates list of all lines to be drawn, size: 1 * Number_of_Lines * 4
    vertices: cordinates numpy array of all vertices, size: 1 * Number_of_Vertices * 2
    """
    y_max, x_max, _ = img_BGR.shape
    # Assign cordinates for the 4 corners
    Point_Lower_Left = (round(0.05 * x_max), y_max - 1)
    Point_Lower_Right = (round(0.98 * x_max), y_max - 1)
    Point_Upper_Left = (round(0.45 * x_max), round(0.6 * y_max))
    Point_Upper_Right = (round(0.55 * x_max), round(0.6 * y_max))
    Point_list = [Point_Lower_Left, Point_Lower_Right,
                  Point_Upper_Right, Point_Upper_Left]
    line = []
    vertices = []
    for i in range(len(Point_list)):
        line.append(Point_list[0] + Point_list[1])
        vertices.append(Point_list[0])
        Point_list = Point_list[1:] + Point_list[:1]
    lines = [line]
    vertices = np.array([vertices])
    return vertices, lines


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns the coordinates of Hough lines.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    ##lane_line = hough2lane_lines(lines, img)
    ##line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    ##draw_lines(line_img, lane_line)
    return lines


def hough2lane_lines(hough_line_list, img):
    """
    According to given hough lines, calculate 2 lane lines which averages lane lines on both sides.
    """
    h_img, w_img, _ = img.shape
    kl_min = -10
    kl_max = -0.5
    kr_min = 0.5
    kr_max = 10

    # Group all lines into 2 lists. Each line should be listed together with its k.
    line_list_l = []
    line_list_r = []

    for hough_line in hough_line_list:
        k = (hough_line[0][3] - hough_line[0][1]) / (hough_line[0][2] - hough_line[0][0])
        if k > kl_min and k < kl_max:
            line_list_l.append(hough_line)
        elif k > kr_min and k < kr_max:
            line_list_r.append(hough_line)

    # Average all ks
    k_l = 0
    k_r = 0
    x_list_l = []
    y_list_l = []
    x_list_r = []
    y_list_r = []

    if len(line_list_l) > 0:
        for line in line_list_l:
            x_list_l.append(line[0][0])
            x_list_l.append(line[0][2])
            y_list_l.append(line[0][1])
            y_list_l.append(line[0][3])
            k_l = (max(y_list_l) - min(y_list_l)) / (max(x_list_l) - min(x_list_l))

    if len(line_list_r) > 0:
        for line in line_list_r:
            x_list_r.append(line[0][0])
            x_list_r.append(line[0][2])
            y_list_r.append(line[0][1])
            y_list_r.append(line[0][3])
            k_r = (max(y_list_r) - min(y_list_r)) / (max(x_list_r) - min(x_list_r))

    # Calculate the 2 x cordinates in the 2 lines to be returned
    line_l = [0, h_img - 1, 0, 0.6 * h_img]
    line_r = [w_img - 1, h_img - 1, w_img - 1, 0.6 * h_img]
    if k_l != 0:
        line_l[0] = min(x_list_l) - (line_l[1] - max(y_list_l)) / k_l
        line_l[2] = max(x_list_l) - (line_l[3] - min(y_list_l)) / k_l
    if k_r != 0:
        line_r[0] = min(x_list_r) + (line_r[1] - min(y_list_r)) / k_r
        line_r[2] = max(x_list_r) + (line_r[3] - max(y_list_r)) / k_r
    line_l = [map(int, line_l)]
    line_r = [map(int, line_r)]

    return np.array([line_l, line_r])


if __name__ == "__main__":
    img1 = cv2.imread('test_images/solidWhiteCurve.jpg')
    f_l = FeatureCollector(img1)
    f_l.image_show()
