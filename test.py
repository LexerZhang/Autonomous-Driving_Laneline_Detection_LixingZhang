import cam_image_process.lanes_detector as ld
import cv2

img1 = cv2.imread('test_images/solidWhiteCurve.jpg')
f_l = ld.FeatureCollector(img1)
f_l.add_layer('R_channel','feature')
f_l.layers_dict['R_channel'].channel_selection('R').binary_threshold((50,100))
f_l.add_layer('G_channel','feature')
f_l.layers_dict['G_channel'].channel_selection('G').binary_threshold((50,100))


img_new = f_l('R_channel','G_channel','add')

ld.helper.image_show(img_new)
