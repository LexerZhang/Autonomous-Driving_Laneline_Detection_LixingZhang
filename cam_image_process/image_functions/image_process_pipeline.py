"""Provide basic models to build up image pipelines."""
import cv2
import numpy as np

from . import helper


class Canvas2:
    """
    The superclass for any single-channel image elements.
    :self.attribute canvas: a numpy image matrix of the same size as the image to be processed.
    """

    def __init__(self, img):
        """
        set the self.img as a black image of the same size as img.
        :param img: the feature image numpy matrix
        """
        self.canvas = np.zeros_like(img)

    def calibration(self, camMtx, distCoe):
        """
        TODO: undistort the image using the provided parameters.
        :param camMtx: the camera Matrix
        :param distCoe: the distortion coefficient
        """

    def __and__(self, other):
        """Return the bitwise-and result of 2 image matrices"""
        return helper.image_normalization(np.logical_and(self.canvas, other.canvas))

    def __or__(self, other):
        """Return the bitwise-or result of 2 image matrices"""
        return helper.image_normalization(np.logical_or(self.canvas, other.canvas))

    def __xor__(self, other):
        """Return the bitwise-xor result of 2 images matrices"""
        return helper.image_normalization(np.logical_xor(self.canvas, other.canvas))

    def __add__(self, other):
        """Combine the 2 image features by setting them to 2 color channels."""
        return helper.image_normalization(np.stack((self.canvas, other.canvas, np.zeros_like(self.canvas)), axis=2))


class Canvas3(Canvas2):
    """
    The superclass for any three-channel image elements.
    :self.attribute canvas: an image matrix of the same size as the image to be processed.
    :self.attribute canvas3: the original 3-channel BGR image matrix.
    """
    def Canvas2GRAY(self):
        """Turn the Canvas to gray scale image."""
        self.canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        return self

    def __add__(self, other):
        """Combine the 2 image features by setting them to 2 color channels."""
        if len(self.canvas.shape)==3:self.Canvas2GRAY()
        if len(self.canvas.shape)==3:other.Canvas2GRAY()
        return Canvas2.__add__(self, other)


class ImgFeature2(Canvas2):
    """
    The class takes in a single-channel image matrix and can do multiple openCV operations on the image.
    :self.attribute canvas: the feature image(single channel)
    :self.attribute img: the original image(single channel)
    """

    def __init__(self, img):
        """
        The initialization takes in an image matrix.
        For 2-dim images, format must be GRAY;
        for 3-dim images, format must be BGR.

        :param img: 2-dim image matrix
        """
        Canvas2.__init__(self, img)
        self.img = img

    def binary_threshold(self, thresholds=(0,255)):
        """Create a binary image, in which 0 refers to the region within the thresholds. """
        self.canvas = (self.canvas>thresholds[0])&(self.canvas<thresholds[1])
        return self

    def gaussian_blur(self, sigma, k_size=3):
        """TODO: Use a Gaussian Kernel to blur the image"""

    def sobel_convolute(self, method, k_size=3):
        """TODO: Use a Sobel kernel to calculate the derivative of the image."""


class ImgFeature3(Canvas3, ImgFeature2):
    """
    The class takes in a three-channel image matrix and append some channel generating image_functions
    to the ImageFeature2 superclass.
    :self.attribute img: the feature image(single channel)
    """

    def channel_selection(self, label):
        """
        Get the specified channel image.
        :param label: Supported labels:
                      ('R', 'G', 'B', 'H', 'L', 'S')
        """

        if label in 'BGR':
            self.canvas = self.img[:,:,'BGR'.index(label)]
        elif label in 'HLS':
            self.canvas = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)[:,:,'HLS'.index(label)]
        else:
            print("Sorry but this channel is not supported, return R channel instead.")
        return self

    def gaussian_blur(self, sigma, k_size=3):
        """TODO: Use a Gaussian Kernel to blur the image"""


class ImgMask2(Canvas2):
    """
    Create an binary image mask using different kinds of edge extraction techniques.
    :self.attribute img: an image mask matrix of the same size as the image to be processed.
    """

    def __init__(self, img):
        """TODO: inherits the canvas.__init__, instead generate a white image."""

    def geometrical_mask(self, vertices):
        """
        TODO: mask out the region outside of the vertices.
        :param vertices: numpy matrix of vertices, size: num_vertices x num_edges x 2
        """

    def straight_lines(self, vertices, color=255, thickness=6):
        """TODO: Create a mask with lines drawn with the parameters provided."""


class ImgMask3(Canvas3, ImgMask2):
    """TODO: image mask in 3 channels"""


class FeatureCollector:
    """
    Collects a list of features extracted from a single image.
    Use them for showing, combination, or simply acts as a pipeline.
    :self.attribute img: the BGR or GRAY image matrix
    :self.attribute layers_dict: list of image_feature instance
    :self.attribute color_model: the color model of image
    :self.attribute cameraMtx: camera matrix for calibration
    :self.attribute dist_coef: distortion coefficients for calibration
    """

    def __init__(self, img, color_model='BGR'):
        """
        The initialization takes in an image matrix.
        Acceptable formats including:
            GRAY scale
            all validate color formats supported by openCV
        Images would be in default stored as uint8 format in BGR or GRAY.
        If the format is not BGR for a 3-dim image, a [format] must be assigned.

        :param img: 2-dim or 3-dim image matrix
        :param color_model: labels among: BAYER_BG, HLS, HSV, LAB, RGB, BGR, GRAY...
        """

        self.img = helper.image_normalization(img)
        self.layers_dict = {}
        if len(self.img.shape) == 2:
            self.color_model = 'GRAY'
        elif color_model != 'BGR':
            l_valid_color_format = [key for key in cv2.__dict__.keys()
                                    if key.startswith('COLOR')
                                    and key.endswith('2BGR')
                                    and len(key.split('_')) == 2]
            if color_model in l_valid_color_format:
                cvt_method = "cv2.COLOR_" + color_model + "2BGR"
                self.img = cv2.cvtColor(self.img, eval(cvt_method))
            else:
                print('Unknown color model, please manually transfer to BGR.')
        self.color_model = 'BGR'

    def add_layer(self, key='layer', type='feature', layer = None):
        """Add a new key:ImgFeature/ImgMask instance to the self.layers_dict."""
        if key == 'layer':
            key = 'layer_' + str(len(self.layers_dict))
        if layer is not None:
            self.layers_dict[key] = layer
        else:
            if type=='feature':
                if self.color_model == "GRAY":
                    self.layers_dict[key] = ImgFeature2(self.img)
                else:
                    self.layers_dict[key] = ImgFeature3(self.img)
            else:
                if self.color_model == "GRAY":
                    self.layers_dict[key] = ImgFeature2(self.img)
                else:
                    self.layers_dict[key] = ImgFeature3(self.img)

    def get_chessboard_calibrators(self, chessboard_img, ):
        """
        TODO: get calibrators using a chessboard image.
        :param chessboard_img:
        """

    def image_show(self):
        helper.image_show(self.img)

    def __call__(self, key1, key2, method='and'):
        """
        Return the Combination of 2 features in the self.layers_dict according to the method.
        :param key1, key2: The keys of canvases to be combined.
        :param method: Choose from("and", "or", "xor", "add")
        """
        try:
            layer1 = self.layers_dict[key1]
            layer2 = self.layers_dict[key2]
        except:
            print("Invalid keys!")
            return
        if method == 'and': return layer1&layer2
        elif method == 'or': return layer1|layer2
        elif method == 'xor': return layer1^layer2
        elif method == 'add': return layer1+layer2
        else:
            print("Doesn't support such method, sorry.")
            return


def region_of_interest(img, vertices):
    """
    Creates an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def line_vertices(img_BGR):
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
    return (lines, vertices)


def sobel_mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255), dir_thresh=(0, np.pi / 2)):
    """Use Sobel kernels to calculate the magnitude&direction of derivatives of an image.

    :input: img: image object RGB/GRAY
    :input: mag_thresh: tuple of magnitude thresholds
    :input: dir_thresh: tuple of direction thresholds
    :input: sobel_kernel: size of the sobel kernel

    :output: output_img: pixels where the sobel magnitude and direction both fall in their thresholds.
    """
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif len(img.shape) == 2:
        img_gray = cv2.copy(img)
    dx_img_sobel = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    dy_img_sobel = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    d_img_mag = np.sqrt(np.square(dx_img_sobel) + np.square(dy_img_sobel))
    d_img_dir = np.arctan2(np.absolute(dy_img_sobel), np.absolute(dx_img_sobel))
    d_img_mag_scaled = np.uint8(255 * d_img_mag / d_img_mag.max())
    d_img_mag_bin = (d_img_mag_scaled > mag_thresh[0]) & (d_img_mag_scaled < mag_thresh[1])
    d_img_dir_bin = (d_img_dir > dir_thresh[0]) & (d_img_dir < dir_thresh[1])
    return np.uint8(d_img_mag_bin & d_img_dir_bin) * 255


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
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

    # To Do: Calculate the 2 x cordinates in the 2 lines to be returned
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
