"""
This file contains a pipeline for lane detection with sliding windows.
"""

from . import parameters as prm
from .image_functions.image_process_pipeline import *


def image_preprocess_pipeline(fc, img_raw):
    """
    Preprocess steps to extract edges.
    :param fc: A FeatureCollector Instance.
    :return: fc
    """
    vertices, _ = helper.get_vertices(fc.img)
    # fc.image_show()  # The original image

    # The trapezoid region mask
    region_mask = ImgMask3(img_raw)
    region_mask.geometrical_mask(vertices, ignore_mask_color=(255, 255, 255))
    fc.add_layer("region_mask", "mask", region_mask)

    # The sobel magnitude binary
    sobel_mag_s = ImgFeature3(img_raw)
    sobel_mag_s.channel_selection('S').gaussian_blur(k_size=prm.gaussian_kernel_size)
    sobel_mag_s.sobel_convolute("mag").binary_threshold(prm.canny_mag_trs)
    fc.add_layer("sobel_mag_s", layer=sobel_mag_s)

    # The sobel direction binary
    sobel_dir_s = ImgFeature3(img_raw)
    sobel_dir_s.channel_selection('S').gaussian_blur(k_size=prm.gaussian_kernel_size)
    sobel_dir_s.sobel_convolute("dir").binary_threshold(prm.canny_dir_trs)
    fc.add_layer("sobel_dir_s", layer=sobel_dir_s)

    # The warped original
    fc.warp().image_show()
    # fc.image_save("color_warped",path=prm.output_img_path)

    # The warped binary
    fc.combine("sobel_mag_s", "sobel_dir_s", "and")
    fc.combine("sobel_mag_s", "region_mask", "and")
    fc.warp("img_processed").image_show()
    # fc.image_save("binary_warped",path=prm.output_img_path)
    return fc


def window_lane_detector_initial(img_raw, degree_of_poly=2):
    """
    Takes a raw image as input and return the processed image, together with the curve parameters.
    :param img_raw: numpy.array of size hxwx3, the raw BGR image of the road.
    :param degree_of_poly: nint, the degree of polynomial curves of the lane lines
    :return img_processed: numpy.array of size hxwx3, the processed with lane lines drawn onto it.
    :return poly_params: numpy.array of size 2x(1+degree_of_poly), the polinomial parameters of both lane lines
    """
    # Step 1: Get the warping parameters and set up the container
    vertices, lines = helper.get_vertices(img_raw)
    y_m, x_m, _ = img_raw.shape
    margin = 10
    vertices = np.float32(vertices)
    dst = np.array([[margin, y_m - margin], [x_m - margin, y_m - margin], [x_m - margin, margin], [margin, margin]],
                   dtype=np.float32)
    warp_mtx = cv2.getPerspectiveTransform(vertices, dst)
    fc = FeatureCollector(img_raw, calibrators=(1, 0, 0, warp_mtx))
    edge_lines = ImgMask3(img_raw)
    edge_lines.straight_lines(lines)
    fc.add_layer("edge_lines", "mask", edge_lines)
    fc.combine("main", "edge_lines", "mix")

    # Step2: Edge Extracting pipeline
    fc = image_preprocess_pipeline(fc, img_raw)

    ### Step 3: Use a sliding window to detect lane lines
    lane_curves_initial = ImgMask3(fc.img_processed)
    squares_list, left_lane_params, right_lane_params = lane_finding_sliding_windows(fc.img_processed[:, :, 0],
                                                                                     degree_of_poly)
    # lane_curves_initial.save_layer("lane_curves_warped", path=prm.output_img_path)
    points_l = get_points(lane_curves_initial.canvas, left_lane_params)
    points_r = get_points(lane_curves_initial.canvas, right_lane_params)
    points_all = np.vstack((points_l, points_r[::-1, :]))
    points_list = np.array((points_all,))
    lane_curves_initial.geometrical_mask(points_list, (0, 50, 0))
    lane_curves_initial.curves(left_lane_params, (255, 0, 0), thickness=prm.thickness_of_line).curves(
        right_lane_params, (0, 0, 255), thickness=prm.thickness_of_line, show_key=True)

    fc.add_layer("lane_curves_initial", layer=lane_curves_initial)
    fc.warp("lane_curves_initial", reverse=True)
    fc.combine("main", "lane_curves_initial", "mix", (1, 1, 0))
    fc.image_show()
    # fc.image_save("lane_curves_drawn", path=prm.output_img_path)
    return left_lane_params, right_lane_params, fc.img_processed, fc

    # Step 4: Use the detected lane line as base to find sequential lane lines.


def window_lane_detector_sequential(fc, img_raw, left_lane_params=0, right_lane_params=0, degree_of_poly=2):
    """
    The sequential lane detector, must have non-zero lane parameters as input.
    :param fc: A feature collector instance, with calibrators initialized
    :param img_raw: numpy.array of size hxwx3, the raw BGR image of the road.
    :param left_lane_params: The polynomial parameters of the left lane.
    :param right_lane_params: The polynomial parameters of the right lane.
    :param degree_of_poly: nint, the degree of polynomial curves of the lane lines
    :return img_processed: numpy.array of size hxwx3, the processed with lane lines drawn onto it.
    :return poly_params: numpy.array of size 2x(1+degree_of_poly), the polinomial parameters of both lane lines
    """
    # Step 1: Reload the FeatureCollector
    if left_lane_params is 0 and right_lane_params is 0:
        print("Not Intialized. Aborting")
        return
    fc.image_reload(img_raw)

    # Step 2: Preprocssing Pipeline
    fc = image_preprocess_pipeline(fc, img_raw)

    # Step 3: Use previous curves to detect lane lines
    lane_curves_sequential = ImgMask3(fc.img_processed)
    central_line_l = get_points(img_raw, left_lane_params)
    left_vertices, left_lane_inds = lane_segments_in_serpent(fc.img_processed[:, :, 0], central_line_l, prm.margin_pix)
    central_line_r = get_points(img_raw, right_lane_params)
    right_vertices, right_lane_inds = lane_segments_in_serpent(fc.img_processed[:, :, 0], central_line_r,
                                                               prm.margin_pix)
    serpent_vertices = np.array((left_vertices, right_vertices))

    leftx = left_lane_inds[1, :]
    lefty = left_lane_inds[0, :]
    rightx = right_lane_inds[1, :]
    righty = right_lane_inds[0, :]
    indices_mask = ImgMask3(fc.img_processed)
    indices_mask.fill_region(lefty, leftx, (0, 0, 255)).fill_region(righty, rightx, (255, 0, 0)).polylines(
        serpent_vertices, True, (255, 255, 0), show_key=True)
    # indices_mask.save_layer("window_detection", path=prm.output_img_path)
    left_lane_params = np.polyfit(lefty, leftx, degree_of_poly)
    right_lane_params = np.polyfit(righty, rightx, degree_of_poly)
    # lane_curves_sequential.save_layer("lane_curves_warped", path=prm.output_img_path)
    points_l = get_points(lane_curves_sequential.canvas, left_lane_params)
    points_r = get_points(lane_curves_sequential.canvas, right_lane_params)
    points_all = np.vstack((points_l, points_r[::-1, :]))
    points_list = np.array((points_all,))
    lane_curves_sequential.geometrical_mask(points_list, (0, 50, 0))
    lane_curves_sequential.curves(left_lane_params, (255, 0, 0), thickness=prm.thickness_of_line).curves(
        right_lane_params, (0, 0, 255), thickness=prm.thickness_of_line, show_key=True)

    fc.add_layer("lane_curves_sequential", layer=lane_curves_sequential)
    fc.warp("lane_curves_sequential", reverse=True)
    fc.combine("main", "lane_curves_sequential", "mix", (1, 1, 0))
    fc.image_show()
    # fc.image_save("lane_curves_drawn", path=prm.output_img_path)
    return left_lane_params, right_lane_params, fc.img_processed, fc


def lane_finding_sliding_windows(img_binary_warped, degree_of_poly=2):
    """
    Calculate both lane line curves' parameters.
    :param img_binary_warped: The warped binary image of road, numpy.array[h, w]
    :param degree_of_poly: degrees of curve to be calculated
    :return: left_fit, tuple(degree_of_curves + 1)
    :return: right_fit, tuple(degree_of_curves + 1)
    """
    left_lane_inds = np.zeros((2, 0), dtype=np.int32)
    right_lane_inds = np.zeros_like(left_lane_inds, dtype=np.int32)
    squares_list = []

    histogram = np.sum(img_binary_warped[img_binary_warped.shape[0] // 2:, :], axis=0)
    middle_x = np.int(histogram.shape[0] / 2)
    leftx_center = np.argmax(histogram[:middle_x])
    rightx_center = np.argmax(histogram[middle_x:]) + middle_x

    win_height = img_binary_warped.shape[0] // prm.num_windows
    for i in range(prm.num_windows):
        win_y_low = img_binary_warped.shape[0] - i * win_height
        win_y_high = img_binary_warped.shape[0] - (i + 1) * win_height

        squares_list.append(get_window_edges(win_y_low, win_y_high, leftx_center, prm.margin_pix))
        squares_list.append(get_window_edges(win_y_low, win_y_high, rightx_center, prm.margin_pix))

        good_left_inds = lane_segments_in_slide(img_binary_warped, win_y_low, win_y_high,
                                                leftx_center, prm.margin_pix)
        good_right_inds = lane_segments_in_slide(img_binary_warped, win_y_low, win_y_high,
                                                 rightx_center, prm.margin_pix)
        left_lane_inds = np.concatenate((left_lane_inds, good_left_inds), axis=1)
        right_lane_inds = np.concatenate((right_lane_inds, good_right_inds), axis=1)
        if (good_left_inds.shape[1] > prm.min_pix_4_update):
            leftx_center = np.int(good_left_inds[1, :].mean())
        if (good_right_inds.shape[1] > prm.min_pix_4_update):
            rightx_center = np.int(good_right_inds[1, :].mean())

    leftx = left_lane_inds[1, :]
    lefty = left_lane_inds[0, :]
    rightx = right_lane_inds[1, :]
    righty = right_lane_inds[0, :]
    indices_mask = ImgMask3(cv2.cvtColor(img_binary_warped, cv2.COLOR_GRAY2BGR))
    indices_mask.fill_region(lefty, leftx, (0, 0, 255)).fill_region(righty, rightx, (255, 0, 0)).straight_lines(
        squares_list,
        (255, 255, 0), 2,
        True)
    # indices_mask.save_layer("window_detection", path=prm.output_img_path)
    left_fit = np.polyfit(lefty, leftx, degree_of_poly)
    right_fit = np.polyfit(righty, rightx, degree_of_poly)
    return squares_list, left_fit, right_fit


def get_window_edges(win_y_low, win_y_high, x_center, margin):
    edges_list = []
    edges_list.append((x_center - margin, win_y_low, x_center + margin, win_y_low))
    edges_list.append((x_center + margin, win_y_low, x_center + margin, win_y_high))
    edges_list.append((x_center + margin, win_y_high, x_center - margin, win_y_high))
    edges_list.append((x_center - margin, win_y_high, x_center - margin, win_y_low))
    return edges_list


def get_points(img, params):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0] // 10, dtype=np.int32)
    plotx = np.zeros_like(ploty, dtype=np.float64)
    for param in params:
        plotx *= ploty
        plotx += param
    plotx = np.array(plotx, dtype=np.int32)
    points = np.array((plotx, ploty)).T
    return points


def lane_segments_in_slide(img_binary_warped, win_y_low, win_y_high, x_center, margin):
    win_x_left = x_center - margin
    win_x_right = x_center + margin
    mask = np.zeros_like(img_binary_warped)
    mask[win_y_high:win_y_low, win_x_left:win_x_right] = 1
    window = cv2.bitwise_and(img_binary_warped, mask)
    good_points = np.array(window.nonzero())
    return good_points


def lane_segments_in_serpent(img_binary_warped, central_line, margin):
    mask_layer = ImgMask2(img_binary_warped)
    margin_array = np.array([margin, 0])
    left_line = central_line - margin_array
    right_line = central_line + margin_array
    vertices = np.vstack((left_line, right_line[::-1, :]))
    vertices = np.array((vertices,))
    mask_layer.geometrical_mask(vertices)
    window = cv2.bitwise_and(img_binary_warped, mask_layer.canvas)
    good_points = np.array(window.nonzero())
    return vertices[0], good_points
