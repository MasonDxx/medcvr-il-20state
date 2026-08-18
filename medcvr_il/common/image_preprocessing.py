from __future__ import annotations

import cv2
import numpy as np

from medcvr_il.common.config_schema import ImagePreprocessorConfig, TopicConfig


def crop_image(
    img: np.ndarray,
    out_w: int,
    out_h: int,
    x_align: str = 'center',
    y_align: str = 'center',
) -> np.ndarray:
    """
    Crop an image to (out_w, out_h) using alignment rules.
    Args:
        img: Input image (H, W, C)
        out_w: Target width
        out_h: Target height
        x_align: Horizontal alignment ('left', 'center', 'right')
        y_align: Vertical alignment ('top', 'center', 'bottom')
    Returns:
        Cropped image (out_h, out_w, C)
    Raises:
        ValueError: If requested crop exceeds image dimensions
    """
    h, w = img.shape[:2]
    if out_w > w or out_h > h:
        raise ValueError(f"Requested crop {out_w}x{out_h} exceeds image size {w}x{h}")

    if x_align == 'left':
        x0 = 0
    elif x_align == 'right':
        x0 = w - out_w
    else: 
        x0 = (w - out_w) // 2

    if y_align == 'top':
        y0 = 0
    elif y_align == 'bottom':
        y0 = h - out_h
    else:
        y0 = (h - out_h) // 2

    x0 = max(0, min(x0, w - out_w))
    y0 = max(0, min(y0, h - out_h))
    return img[y0:y0+out_h, x0:x0+out_w]


def resize_image(img: np.ndarray, resize_hw: tuple[int, int] | list[int] | None) -> np.ndarray:
    """
    Resize an image to (height, width). Defaults to (240, 320).
    """
    if resize_hw is None:
        return img

    resize_h, resize_w = int(resize_hw[0]), int(resize_hw[1])
    return cv2.resize(img, (resize_w, resize_h))


def preprocess_image(
    img: np.ndarray,
    config: ImagePreprocessorConfig | None = None,
) -> np.ndarray:
    """
    Apply crop (with alignment) followed by resize, then return CHW.
    """
    crop_hw = config.rectangular_crop_hw if config is not None else None
    resize_hw = config.image_resize_hw if config is not None else None
    align = config.align if config is not None else None

    if crop_hw is not None:
        crop_h, crop_w = int(crop_hw[0]), int(crop_hw[1])
        x_align, y_align = ('center', 'center')
        if align is not None:
            x_align, y_align = align
        img = crop_image(img, out_w=crop_w, out_h=crop_h, x_align=x_align, y_align=y_align)
    img = resize_image(img, resize_hw)
    return np.transpose(img, (2, 0, 1))


def resolve_camera_preprocessor_configs(
    cam_names: list[str],
    topics: dict[str, TopicConfig],
    global_pre: ImagePreprocessorConfig | None,
) -> dict[str, ImagePreprocessorConfig]:
    """Resolve final preprocessor config per camera.

    Priority is camera-specific preprocessor object first, then global preprocessor object.
    If neither is present, use defaults.
    """
    resolved: dict[str, ImagePreprocessorConfig] = {}
    for cam_name in cam_names:
        topic_cfg = topics.get(cam_name)
        topic_pre = topic_cfg.pre_processor if topic_cfg is not None else None
        if topic_pre is not None:
            selected = topic_pre
        elif global_pre is not None:
            selected = global_pre
        else:
            selected = ImagePreprocessorConfig()

        resolved[cam_name] = ImagePreprocessorConfig(
            rectangular_crop_hw=selected.rectangular_crop_hw,
            image_resize_hw=selected.image_resize_hw,
            align=selected.align,
        )

    return resolved
