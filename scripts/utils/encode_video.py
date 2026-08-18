import os
import cv2

def encode_video(input_path, output_path):
    try:
        # Read the video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error opening video file: {input_path}")
            return

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'X264')  # Use 'X264' for H.264 encoding
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
        print(f"Encoded video saved to: {output_path}")

    except Exception as e:
        print(f"Error processing file {input_path}: {e}")

def main():
    input_folder = os.path.join('diffusion_policy', 'data', 'push_random_block2', 'videos', '28')
    output_folder = os.path.join('diffusion_policy','data', 'push_random_block2', 'videos_encoded', '28')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for episode in os.listdir(input_folder):
        episode_folder = os.path.join(input_folder, episode)
        if os.path.isdir(episode_folder):
            input_path = os.path.join(episode_folder, '1.mp4')
            episode_output_folder = os.path.join(output_folder, episode)
            if not os.path.exists(episode_output_folder):
                os.makedirs(episode_output_folder)
            output_path = os.path.join(episode_output_folder, '1.mp4')
            
            encode_video(input_path, output_path)

if __name__ == "__main__":
    main()