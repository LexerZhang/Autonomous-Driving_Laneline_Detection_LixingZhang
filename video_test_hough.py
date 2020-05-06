from cam_image_process import lanes_detector_Hough_Transform as ld
from moviepy.editor import VideoFileClip

white_output = "./test_videos_output/test_output.mp4"
clip1 = VideoFileClip("./test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(ld.Hough_image_lane_detector)
white_clip.write_videofile(white_output, audio=False)