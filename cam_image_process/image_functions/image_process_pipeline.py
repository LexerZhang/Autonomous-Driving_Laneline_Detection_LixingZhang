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
        self.canvas = 255*np.ones_like(img[:,:,0])
        #self.img_normalization()

    def show_layer(self, key=False):
        if key:
            helper.image_show(self.canvas)

    def img_normalization(self):
        self.canvas=helper.image_normalization(self.canvas)

    def __and__(self, other):
        """Return the bitwise-and result of 2 image matrices"""
        self.canvas = helper.image_normalization(cv2.bitwise_and(self.canvas, other.canvas))
        return self.canvas

    def __or__(self, other):
        """Return the bitwise-or result of 2 image matrices"""
        self.canvas = helper.image_normalization(cv2.bitwise_or(self.canvas, other.canvas))
        return self.canvas

    def __xor__(self, other):
        """Return the bitwise-xor result of 2 images matrices"""
        self.canvas = helper.image_normalization(cv2.bitwise_xor(self.canvas, other.canvas))
        return self.canvas

    def __add__(self, other):
        """Combine the 2 image features by setting them to 2 color channels."""
        self.canvas = helper.image_normalization(np.stack((self.canvas, other.canvas, np.zeros_like(self.canvas)), axis=2))
        return self.canvas


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
        self.canvas = img.copy()

    def binary_threshold(self, thresholds=(0, 255), show_key=False):
        """Create a binary image, in which 0 refers to the region within the thresholds. """
        self.canvas = helper.image_normalization((self.canvas>thresholds[0])&(self.canvas<thresholds[1]))
        self.show_layer(show_key)
        return self

    def gaussian_blur(self, sigma, k_size=(3,3), show_key=False):
        """Use a Gaussian Kernel to blur the image"""
        self.canvas = cv2.GaussianBlur(self.canvas, k_size, sigma)
        self.show_layer(show_key)
        return self


    def sobel_convolute(self, method, k_size=3, show_key=False):
        """
        Use a Sobel kernel to calculate the derivative of the image.
        """
        img_gray = self.canvas
        if method == 'x':
            self.canvas = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k_size))
        elif method == 'y':
            self.canvas = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k_size))
        elif method == 'dir':
            dx_img_sobel = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k_size))
            dy_img_sobel = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k_size))
            self.canvas = np.arctan2(np.absolute(dy_img_sobel), np.absolute(dx_img_sobel))
        else:
            dx_img_sobel = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k_size))
            dy_img_sobel = np.absolute(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k_size))
            self.canvas = np.sqrt(np.square(dx_img_sobel) + np.square(dy_img_sobel))
        #self.img_normalization()
        self.show_layer(show_key)
        return self


class ImgFeature3(Canvas3, ImgFeature2):
    """
    The class takes in a three-channel image matrix and append some channel generating image_functions
    to the ImageFeature2 superclass.
    :self.attribute img: the feature image(single channel)
    """

    def channel_selection(self, label, show_key=False):
        """
        Get the specified channel image.
        :param label: Supported labels:
                      ('R', 'G', 'B', 'H', 'L', 'S')
        """
        if label.upper() == 'GRAY':
            self.canvas = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        elif label.upper() in 'BGR':
            self.canvas = self.img[:,:,'BGR'.index(label)]
        elif label.upper() in 'HLS':
            self.canvas = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)[:,:,'HLS'.index(label)]
        else:
            print("Sorry but this channel is not supported, return GRAY Scale instead.")
            self.canvas = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_normalization()
        self.show_layer(show_key)
        return self

    def sobel_convolute(self, method, k_size=3,show_key=False):
        """In case the canvas is still 3-channeled, convert to GRAY_Scale first."""
        if self.canvas.shape == 3:
            self.Canvas2GRAY()
        return ImgFeature2.sobel_convolute(self, method, k_size,show_key)


class ImgMask2(Canvas2):
    """
    Create an binary image mask using different kinds of edge extraction techniques.
    :self.attribute img: an image mask matrix of the same size as the image to be processed.
    """

    def __init__(self, img):
        """inherits the canvas.__init__, instead generate a white image."""
        Canvas2.__init__(self, img)
        self.img = img

    def geometrical_mask(self, vertices, show_key=False):
        """
        mask out the region outside of the vertices.
        :param vertices: numpy matrix of vertices, size: num_vertices x num_edges x 2
        """
        mask = np.zeros_like(self.canvas) # defining a blank mask to start with
        ignore_mask_color = 255
        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        self.canvas = mask
        if show_key:
            self.show_layer(show_key)
        return self

    def straight_lines(self, vertices, color=(0, 0, 255), thickness=6, show_key=False):
        """Create a mask with lines drawn with the parameters provided."""
        self.canvas = helper.draw_lines(self.canvas, vertices,color,thickness)
        if show_key:
            self.show_layer(show_key)


class ImgMask3(Canvas3, ImgMask2):
    """TODO: image mask in 3 channels"""
    def __init__(self, img):
        ImgMask2.__init__(self, img)

    def geometrical_mask(self, vertices, show_key=False):
        """
        mask out the region outside of the vertices.
        :param vertices: numpy matrix of vertices, size: num_vertices x num_edges x 2
        """
        mask = np.zeros_like(self.img) # defining a blank mask to start with
        if len(self.img.shape) > 2:
            channel_count = self.img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        self.canvas = mask
        self.show_layer(show_key)
        return self


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

    def __init__(self, img, color_model='BGR', calibrators=(0,0,0,0)):
        """
        The initialization takes in an image matrix.
        Acceptable formats including:
            GRAY scale
            all validate color formats supported by openCV
        Images would be in default stored as uint8 format in BGR or GRAY.
        If the format is not BGR for a 3-dim image, a [format] must be assigned.

        :param img: 2-dim or 3-dim image matrix
        :param color_model: labels among: BAYER_BG, HLS, HSV, LAB, RGB, BGR, GRAY...
        :param calibrators: calibration parameters list following the order(number of
            chessboard images fed, Camera Matrix, Distortion Coefficient, Warp Matrix)
        """

        self.img = helper.image_normalization(img)
        self.img_processed = self.img.copy()
        self.layers_dict = {}
        self.add_layer('main','mask')
        self.calibrators = {"number_of_img":calibrators[0], "CamMtx":calibrators[1],
                            "DistCoe":calibrators[2], "WarpMtx":calibrators[3]}
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
                self.layers_dict[key] = ImgMask2(self.img)

    def get_chessboard_calibrators(self, chessboard_img, num_x,num_y=(2,2)):
        """
        Get calibrators using a chessboard image and the specified number of corners.
        :param chessboard_img: A chess board image
        :param corners_number: The number of corners on chessboard in x and y directions
        """
        obj_points = []
        img_points = []
        objp = np.zeros((num_x*num_y,3),np.float32)
        objp[:,:2] = np.mgrid[0:num_x, 0:num_y].T.reshape(-1,2)
        chessboard_img_gray = cv2.cvtColor(chessboard_img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(chessboard_img_gray, (num_x, num_y))
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
            ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points,img_points,
                                                       chessboard_img_gray.shape[::-1],None,None)
            self.calibrators["number_of_img"] += 1
            n = self.calibrators["number_of_img"]
            self.calibrators["CamMtx"] = (self.calibrators["CamMtx"] * (n-1)+mtx)/n
            self.calibrators["DistCoe"] = (self.calibrators["DistCoe"] * (n-1)+dist)/n
            chessboard_undistorted_gray = cv2.undistort(chessboard_img_gray, self.calibrators["CamMtx"],
                                                        self.calibrators["DistCoe"])
            _, points = cv2.findChessboardCorners(chessboard_undistorted_gray, (num_x, num_y))
            src_left_up = points[0][0]
            src_right_up = points[num_x-1][0]
            src_left_low = points[num_x*(num_y-1)][0]
            src_right_low = points[num_x*num_y-1][0]
            dst_left_up = [100, 100]
            dst_left_low = [100, chessboard_undistorted_gray.shape[0] - 100]
            dst_right_up = [chessboard_undistorted_gray.shape[1] - 100, 100]
            dst_right_low = [chessboard_undistorted_gray.shape[1] - 100, chessboard_undistorted_gray.shape[0] - 100]
            src = np.array([src_left_up, src_right_up, src_left_low, src_right_low], dtype=np.float32)
            dst = np.array([dst_left_up, dst_right_up, dst_left_low, dst_right_low], dtype=np.float32)
            self.calibrators["WarpMtx"] = cv2.getPerspectiveTransform(src, dst)
        else:
            print("Unable to detect corners, please try again!")

    def undistort(self):
        """
        Undistort the image using the provided parameters.
        :param camMtx: the camera Matrix
        :param distCoe: the distortion coefficient
        """
        self.img = cv2.undistort(self.img, self.calibrators["CamMtx"], self.calibrators["DistCoe"])
        return self

    def warp(self):
        """
        Warp the image using a perspective transformation matrix.
        :param M: the warp Matrix
        """
        self.img = cv2.warpPerspective(self.img, self.calibrators["WarpMtx"], self.img.shape[1::-1])
        return self

    def image_show(self,show_key=True):
        if show_key:
            helper.image_show(self.img_processed)

    def combine(self, key1, key2, method='and'):
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
        if method == 'and': self.img_processed = layer1&layer2
        elif method == 'or': self.img_processed = layer1|layer2
        elif method == 'xor': self.img_processed = layer1^layer2
        elif method == 'add': self.img_processed = layer1+layer2
        else:
            print("Doesn't support such method, sorry.")
            return
