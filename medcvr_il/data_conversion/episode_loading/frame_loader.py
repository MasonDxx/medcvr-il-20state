from __future__ import annotations

import cv2
import re
from pathlib import Path

import numpy as np
from PIL import Image

from medcvr_il.common.config_schema import ImagePreprocessorConfig
from medcvr_il.common.image_preprocessing import preprocess_image


def natural_key(p: Path):
    """Sort paths naturally so image10 comes after image2."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', p.stem)]


def get_cam_dirs(ep: Path, camera_filter: list = None, cam_names: list = None):
    """Discover camera folders in episode directory.

    Args:
        ep: Episode directory path
        camera_filter: Optional list of camera numbers to include (e.g., [0, 2] for cam0 and cam2)
        cam_names: Optional list of custom camera folder names (e.g., ['static_cam_left', 'static_cam_right'])
                   If provided, uses these names instead of auto-discovering cam0, cam1, ...
    """
    if cam_names:
        cam_dirs = []
        for name in cam_names:
            cam_path = ep / name
            if cam_path.exists() and cam_path.is_dir():
                cam_dirs.append(cam_path)
            else:
                raise FileNotFoundError(f"{ep}: camera folder '{name}' not found")

        if camera_filter is not None:
            cam_dirs = [cam_dirs[i] for i in camera_filter if i < len(cam_dirs)]
    else:
        cam_dirs = sorted(
            [p for p in ep.iterdir() if p.is_dir() and re.match(r'^cam\d+$', p.name)],
            key=lambda x: int(re.search(r'\d+', x.name).group())
        )

        if camera_filter is not None:
            cam_dirs = [p for p in cam_dirs if int(re.search(r'\d+', p.name).group()) in camera_filter]

    if not cam_dirs:
        if camera_filter:
            raise FileNotFoundError(f"{ep}: no cameras matching filter {camera_filter} found")
        elif cam_names:
            raise FileNotFoundError(f"{ep}: none of the specified camera names {cam_names} found")
        else:
            raise FileNotFoundError(f"{ep}: no cam0, cam1, ... directories found")
    return cam_dirs

def load_pngs(folder: Path, image_config: ImagePreprocessorConfig | None = None):
    """Load PNG images and apply optional crop / resize preprocessing."""
    files = sorted(folder.glob("*.png"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"{folder}: no pngs")

    frames = []
    for file_path in files:
        image = np.array(Image.open(file_path).convert("RGB"))
        frames.append(preprocess_image(image, config=image_config))
    return frames



def load_jpgs(folder: Path, image_config: ImagePreprocessorConfig | None = None):
    """Load JPG images and apply optional crop / resize preprocessing."""
    files = sorted(folder.glob("*.jpg"), key=natural_key)
    if not files:
        raise FileNotFoundError(f"{folder}: no jpgs")

    frames = []
    for file_path in files:
        image = cv2.cvtColor(cv2.imread(str(file_path)), cv2.COLOR_BGR2RGB)
        frames.append(preprocess_image(image, config=image_config))
    return frames



def load_mp4(folder: Path, image_config: ImagePreprocessorConfig | None = None):
    """Load MP4 video and apply optional crop / resize preprocessing."""
    vid_path = (folder / "mp4" / "video.mp4") if (folder / "mp4" / "video.mp4").exists() else (folder / "video.mp4")
    if not vid_path.exists():
        raise FileNotFoundError(f"{vid_path} not found")

    cap = cv2.VideoCapture(str(vid_path))
    frames = []

    ok, frame = cap.read()
    while ok:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(preprocess_image(image, config=image_config))
        ok, frame = cap.read()

    cap.release()

    if not frames:
        raise RuntimeError(f"{vid_path}: zero frames")
    return frames


IMAGE_LOADERS_DICT = {"png": load_pngs, "jpg": load_jpgs, "mp4": load_mp4}
