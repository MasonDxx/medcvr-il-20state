from __future__ import annotations

from medcvr_il.common.config_schema import (
    ActionSourceConfig,
    ImagePreprocessorConfig,
    TopicType,
    get_action_dim,
    get_state_dim,
)


def build_dataset_schema(
    cam_configs: dict[str, ImagePreprocessorConfig],
    state_mode: str,
    action_sources: list[ActionSourceConfig],
    state_source_count: int = 1,
):
    """
    Build the LeRobot dataset feature schema.

    Args:
        cam_configs: Pre-resolved dict mapping camera name to its image config.
        state_mode: State observation mode
        action_sources: Action sources to concatenate (length 1 for single action)

    Returns:
        features dict for LeRobotDataset.create()
    """
    C = 3
    state_dim = get_state_dim(state_mode, source_count=state_source_count)
    action_dim = 0
    for source in action_sources:
        jaw_mode_value = source.jaw.value if isinstance(source.jaw, TopicType) else str(source.jaw)
        include_jaw = jaw_mode_value != TopicType.none.value and source.jaw_file is not None
        jaw_value = jaw_mode_value if include_jaw else "none"
        action_mode_value = source.action.value if hasattr(source.action, "value") else str(source.action)
        action_dim += get_action_dim(action_mode_value, jaw_value)

    feats = {}
    for i, cfg in enumerate(cam_configs.values()):
        if cfg.image_resize_hw is None:
            raise ValueError("Camera config must have image_resize_hw set")
        H, W = cfg.image_resize_hw
        feats[f"observation.image.cam{i}"] = {
            "dtype": "image",
            "shape": (C, H, W),
            "names": ["C", "H", "W"]
        }

    feats["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": ["D_state"]
    }
    feats["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": ["D_act"]
    }

    return feats
