from Lane_Detection.cam_image_process import lanes_detector_Hough_Transform as ld
from moviepy.editor import VideoFileClip

input_video_path = "../example_ld_input/test_videos/solidWhiteRight.mp4"
white_output = "../example_ld_output/test_videos_output/test_output.mp4"

if __name__ == "__main__":
    clip1 = VideoFileClip(input_video_path)
    white_clip = clip1.fl_image(ld.Hough_image_lane_detector)
    white_clip.write_videofile(white_output, audio=False)