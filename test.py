import cam_image_process.lanes_detector_Hough_Transform as ld
import cv2
import numpy as np

img1 = cv2.imread('test_images/solidYellowCurve.jpg')
ld.helper.image_show(img1)
img_container = ld.FeatureCollector(img1)
ld.Hough_image_lane_detector(img_container)

#img_chessboard = cv2.imread("./test_images/chessboard_distorted.png")
#f_l = ld.FeatureCollector(img_chessboard)
#f_l.get_chessboard_calibrators(img_chessboard,8,6)
#f_l.undistort()
#ld.helper.image_show(f_l.img)
#f_l.warp()
#ld.helper.image_show(f_l.img)

#layer_test = ld.ImgFeature3(img1)
#layer_test.gaussian_blur(sigma=1, k_size=(11,11), show_key=True)


