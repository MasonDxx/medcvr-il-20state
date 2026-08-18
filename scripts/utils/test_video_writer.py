import cv2

# video_path = "/home/ruthrash/medcvr/data/e4/static_cam/video.mp4"


fps = 30  # frames per second
video_filename ="video.mp4"
image = cv2.imread("442.jpg")  # replace with your image file

frame_size = (image.shape[1], image.shape[0])
codec = cv2.VideoWriter_fourcc(*'mp4v')  # use 'XVID' for .avi, or 'avc1'/'mp4v' for .mp4
video = cv2.VideoWriter(video_filename, codec, fps, frame_size)
video.write(image)  # write the image as a frame
video.release()
video.write(image)  # write the image as a frame
video.write(image)  # write the image as a frame

video.release()

