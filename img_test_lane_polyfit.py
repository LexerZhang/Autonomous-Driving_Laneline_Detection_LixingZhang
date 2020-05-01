import cv2

from cam_image_process import lanes_detection_polyfit as ldsw

# Step1: Input the image and Preview
img1 = cv2.imread("./test_images/test4.jpg")
left_lane_params, right_lane_params, img_processed, fc = ldsw.window_lane_detector_initial(img1)
left_lane_params, right_lane_params, img_processed, fc = ldsw.window_lane_detector_sequential(fc, img1,
                                                                                              left_lane_params,
                                                                                              right_lane_params)
