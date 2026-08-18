import cv2
import numpy as np

def check_video_encoding(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Get the codec used for the video
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])  # Decode FourCC into a readable string

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rec, frame = cap.read()
    if rec:
        print(f"Frame shape: {frame.shape}")
        print(f"Frame dtype: {frame.dtype}")
        print("Pixel Max Value: ", np.amax(frame), " Pixel Min Value: ", np.amin(frame))
    print(f"Video Codec: {codec}")
    print(f"Frame Rate (FPS): {fps}")
    print(f"Resolution: {width}x{height}")
    print(f"Total Frames: {frame_count}")

    cap.release()


if __name__ == "__main__":
    VIDEO_FOLDER_PATH = '/home/medcvr/Downloads/pusht_real/real_pusht_20230105/videos/0/1.mp4'
    # VIDEO_FOLDER_PATH = "/home/medcvr/Amey/medcvr-il/data/big_block_10hz_sec_teleop_discrete/videos/0/1.mp4"
    check_video_encoding(VIDEO_FOLDER_PATH)