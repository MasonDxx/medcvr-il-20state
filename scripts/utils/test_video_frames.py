import cv2

# video_path = "/home/ruthrash/medcvr/data/e4/static_cam/video.mp4"
# video_path = "/home/ruthrash/medcvr/push_block_act_aug_5_2025/e25/static_cam/video.mp4"
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
cap.release()
print(f"Read {count} frames")
