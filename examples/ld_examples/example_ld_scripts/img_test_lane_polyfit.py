"""
This example shows how the polyfit of a lane line is conducted.
"""

import cv2
import os

from Lane_Detection.cam_image_process import lanes_detection_polyfit as ldsw

test_img = "../example_ld_input/test_images/lane_lines/straight_lines1.jpg"
calibration_img_path = "../example_ld_input/ld_camera_cal"
test_img_path = "../example_ld_input/test_images/lane_lines"

# Step1: Input the image and Preview
img1 = cv2.imread(test_img)

# Step2: Initialize the container FeatureCollector object.
name_list = []
cal_imgs_list = []
for path, dir_list, file_list in os.walk(calibration_img_path):
    for name in file_list:
        img_cal = cv2.imread(os.path.join(path, name))
        cal_imgs_list.append(img_cal)
fc = ldsw.container_initialization(img1, cal_imgs_list, show_key=False)

# Step3: Test on all 6 test images
for path, dir_list, file_list in os.walk(test_img_path):
    for name in file_list:
        if not name.startswith("test"): continue
        img_test = cv2.imread(os.path.join(path, name))
        left_lane_params, right_lane_params, fc, _ = ldsw.lane_detector_initial(fc, img_test, 2)
        y_range = [0, fc.img_processed.shape[0]]
        _,_,fc = ldsw.live_info_calculation(left_lane_params, right_lane_params, y_range, fc)
        left_lane_params, right_lane_params, fc, _ = ldsw.lane_detector_sequential(fc, img_test, left_lane_params, right_lane_params)
        # fc.image_save("info_printed_" + name.split('.')[0])

# fc.image_show()