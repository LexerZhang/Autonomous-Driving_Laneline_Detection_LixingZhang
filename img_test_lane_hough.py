from cam_image_process import lanes_detector_Hough_Transform as ld
import cv2

img1 = cv2.imread('test_images/solidWhiteCurve.jpg')
ld.helper.image_show(img1)
img_processed = ld.Hough_image_lane_detector(img1)
ld.helper.image_show(img_processed)

#img_chessboard = cv2.imread("./test_images/chessboard_distorted.png")
#f_l = ld.FeatureCollector(img_chessboard)
#f_l.get_chessboard_calibrators(img_chessboard,8,6)
#f_l.undistort()
#ld.helper.image_show(f_l.img)
#f_l.warp()
#ld.helper.image_show(f_l.img)

#layer_test = ld.ImgFeature3(img1)
#layer_test.gaussian_blur(sigma=1, k_size=(11,11), show_key=True)


