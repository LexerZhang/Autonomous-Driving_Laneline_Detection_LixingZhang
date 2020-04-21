import cam_image_process.lanes_detector as ld
import cv2
import numpy as np

img1 = cv2.imread('test_images/solidWhiteCurve.jpg')
#f_l.add_layer('R_channel','feature')
#f_l.layers_dict['R_channel'].channel_selection('R').binary_threshold((50,100))
#f_l.add_layer('G_channel','feature')
#f_l.layers_dict['G_channel'].channel_selection('G').binary_threshold((50,100))


#img_new = f_l('R_channel','G_channel','add')

#ld.helper.image_show(img_new)

#img_chessboard = cv2.imread("./test_images/chessboard_distorted.png")
#f_l = ld.FeatureCollector(img_chessboard)
#f_l.get_chessboard_calibrators(img_chessboard,8,6)
#f_l.undistort()
#ld.helper.image_show(f_l.img)
#f_l.warp()
#ld.helper.image_show(f_l.img)

#layer_test = ld.ImgFeature3(img1)
#layer_test.gaussian_blur(sigma=1, k_size=(11,11), show_key=True)

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
    return vertices


vertices = line_vertices(img1)
Mask_Test = ld.ImgMask3(img1)
Mask_Test.geometrical_mask(vertices,True)
Layer_Img = ld.ImgFeature3(img1)
img_mixed = Mask_Test & Layer_Img
ld.image_show(img_mixed)
