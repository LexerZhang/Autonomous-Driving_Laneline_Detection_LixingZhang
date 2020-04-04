import numpy as np
# Parameters for the Canny Edge Detector
canny_trs_low = 50
canny_trs_high = 150
gaussian_apertureSize = 3

# Parameters for the Hough Transmitter
rho = 2  # distance resolution in pixels of the Hough grid
theta = np.pi/180  # angular resolution in radians of the Hough grid
threshold = 40     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 100  # minimum number of pixels making up a line
max_line_gap = 150    # maximum gap in pixels between connectable line segments
