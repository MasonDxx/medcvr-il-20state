#!/usr/bin/env python3
"""
Convert raw image episodes to LeRobotDataset format with MP4 support.
Each camera's images are center-cropped and resized to 240×320 (H×W).
Supports PNG, JPG, and MP4 image sources.
"""

from __future__ import annotations
import argparse, os, cv2, re
from pathlib import Path
import os

# ───────── helper for “natural” (numeric) sort ──────────────────────
def natural_key(p: Path):
    """
    Split the path stem into digit / non-digit runs and convert digit runs
    to integers, so e.g. 'image10' > 'image2'.
    """
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', p.stem)]


# ───────── basic helpers ────────────────────────────────────────────
episodes = lambda root: sorted(
    (p for p in root.iterdir() if p.is_dir()),
    key=lambda x: natural_key(x)
)

def cams(ep: Path):
    """Auto-discover cam folders (cam0, cam1, cam2, ...) in episode directory."""
    cam_dirs = sorted(
        [p for p in ep.iterdir() if p.is_dir() and re.match(r'^cam\d+$', p.name)],
        key=lambda x: int(re.search(r'\d+', x.name).group())
    )
    if not cam_dirs:
        raise FileNotFoundError(f"{ep}: no cam0, cam1, ... directories found")
    return cam_dirs


def images_to_video_opencv(image_folder, output_video_path, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images = sorted(images, key=lambda x: natural_key(Path(x))) # Ensure images are processed in a specific order
    # print(images)
    if not images:
        print("No images found in the specified folder.")
        return

    # Read the first image to get dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video saved to {output_video_path}")

# Example usage:
# images_to_video_opencv('path/to/your/images', 'output.mp4', 30)

# -------- main converter -------------------------------------------
def convert(raw: Path, fps: int):
    for ep in episodes(raw):
        print(f"[LeRobot] Converting episode {ep.name}...")
        # Load images from all cams
        cam_dirs = cams(ep)
        all_cam_frames = []
        for cam_dir in cam_dirs:
            print(f"[LeRobot] Processing camera directory: {cam_dir.name}")
            images_to_video_opencv(
                image_folder=cam_dir,
                output_video_path=cam_dir / "video.mp4",
                fps=fps
            )
    print(f"[LeRobot] Data conversion completed.")

   

# -------- CLI -------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",        required=True, type=Path,
                   help="folder containing episode_* sub-directories")
    p.add_argument("--fps",            type=int, default=20)

    args = p.parse_args()

    convert(raw=args.raw_dir,
            fps=args.fps)
