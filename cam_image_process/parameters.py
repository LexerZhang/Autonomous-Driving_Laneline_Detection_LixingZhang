"""
This file serves as settings for the lane detection project.
"""
import numpy as np

# Parameters related to I/O information
output_img_path = "./test_images_output/"
output_video_path = "./test_videos_output/"

# Parameters for the Canny Edge Detector
canny_mag_trs_low = 50
canny_mag_trs_high = 255
canny_mag_trs = (canny_mag_trs_low, canny_mag_trs_high)
canny_dir_trs_low = np.pi*2/9
canny_dir_trs_high = np.pi/3
canny_dir_trs = (canny_dir_trs_low, canny_dir_trs_high)
gaussian_apertureSize = 3
gaussian_kernel_size = (gaussian_apertureSize, gaussian_apertureSize)

# Parameters for the Hough Transmitter
hough_rho = 2  # distance resolution in pixels of the Hough grid
hough_theta = np.pi / 180  # angular resolution in radians of the Hough grid
hough_threshold = 40     # minimum number of votes (intersections in Hough grid cell)
hough_min_line_length = 100  # minimum number of pixels making up a line
hough_max_line_gap = 150    # maximum gap in pixels between connectable line segments

# Parameters for the sliding window transmitter
num_windows = 15
margin_pix = 75
min_pix_4_update = 20
thickness_of_line = 50