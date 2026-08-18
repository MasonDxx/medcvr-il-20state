#!/usr/bin/env python3
"""
Generate a template YAML config covering all fields used by the three main scripts.

  recorder.py          → topics, synchronizer, recorder
  lerobot_eval_ros2.py → topics, synchronizer, inference, global_image_pre_processor
  data_conversion.py   → topics, global_image_pre_processor, data_conversion

Usage:
  uv run python scripts/generate_config.py                         # prints to stdout
  uv run python scripts/generate_config.py -o config/template.yaml
"""

import argparse
import ast
import dataclasses
import inspect
import textwrap
import types
import typing
from dataclasses import MISSING as DC_MISSING
from enum import Enum
from pathlib import Path

import yaml
from omegaconf import MISSING as OC_MISSING

from medcvr_il.common.config_schema import (
    DataConversionConfig,
    ImagePreprocessorConfig,
    ActionOutputConfig,
    InferenceConfig,
    PolicyInferenceConfig,
    RecorderConfig,
    RobotProcessorConfig,
    SynchSubscriberNodeConfig,
    TopicConfig,
    ActionSourceConfig
)


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def get_field_docstrings(cls) -> dict[str, str]:
    """
    Extract PEP 257 attribute docstrings from a dataclass.

    An attribute docstring is a string literal placed immediately after a
    field's annotated assignment, e.g.::

        field_name: type = default
        \"\"\"This text appears in tyro --help and as a YAML comment.\"\"\"

    Returns a mapping of field_name -> docstring text.
    """
    try:
        source = textwrap.dedent(inspect.getsource(cls))
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return {}

    # Find the class node matching cls.__name__ (handles nested classes too)
    class_nodes = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == cls.__name__
    ]
    if not class_nodes:
        return {}

    result = {}
    body = class_nodes[0].body
    for i, stmt in enumerate(body):
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        if i + 1 < len(body):
            next_stmt = body[i + 1]
            if (
                isinstance(next_stmt, ast.Expr)
                and isinstance(next_stmt.value, ast.Constant)
                and isinstance(next_stmt.value.value, str)
            ):
                result[stmt.target.id] = next_stmt.value.value
    return result


# Collect attribute docstrings from every config dataclass so they can be
# injected as YAML comments in the generated template.
_ALL_FIELD_COMMENTS: dict[str, str] = {}
for _cls in [
    TopicConfig,
    ImagePreprocessorConfig,
    SynchSubscriberNodeConfig,
    RecorderConfig,
    InferenceConfig,
    PolicyInferenceConfig,
    ActionOutputConfig,
    RobotProcessorConfig,
    DataConversionConfig,
    ActionSourceConfig
]:
    _ALL_FIELD_COMMENTS.update(get_field_docstrings(_cls))


def _inject_comments(yaml_str: str, field_comments: dict[str, str]) -> str:
    """Prepend ``# <docstring>`` lines before matching YAML field lines."""
    result = []
    for line in yaml_str.splitlines():
        stripped = line.lstrip()
        if stripped and not stripped.startswith('#') and ':' in stripped:
            field_name = stripped.split(':')[0].strip()
            if field_name in field_comments:
                indent = ' ' * (len(line) - len(stripped))
                for cline in field_comments[field_name].strip().splitlines():
                    result.append(f"{indent}# {cline.strip()}")
        result.append(line)
    return '\n'.join(result)


def _field_default(f: dataclasses.Field):
    """Return the effective default for a field, or '???' when MISSING."""
    if f.default is not DC_MISSING and f.default is not OC_MISSING:
        return f.default
    if f.default_factory is not DC_MISSING:
        return f.default_factory()
    return "???"


def _unwrap_optional_dataclass(hint):
    """If hint is `SomeDataclass | None`, return SomeDataclass; else return None."""
    # Python 3.10+ X | None creates types.UnionType
    if isinstance(hint, types.UnionType):
        args = hint.__args__
    elif typing.get_origin(hint) is typing.Union:
        args = typing.get_args(hint)
    else:
        return None
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == 1 and dataclasses.is_dataclass(non_none[0]):
        return non_none[0]
    return None

def _unwrap_list_dataclass(hint):
    """If hint is list[SomeDataclass] or list[SomeDataclass] | None, return SomeDataclass."""
    if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
        args = hint.__args__ if isinstance(hint, types.UnionType) else typing.get_args(hint)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) != 1:
            return None
        hint = non_none[0]

    origin = typing.get_origin(hint)
    if origin is not list:
        return None

    args = typing.get_args(hint)
    if len(args) == 1 and dataclasses.is_dataclass(args[0]):
        return args[0]

    return None


def _unwrap_dict_dataclass(hint):
    """If hint is dict[str, SomeDataclass], return SomeDataclass."""
    origin = typing.get_origin(hint)
    if origin is not dict:
        return None

    args = typing.get_args(hint)
    if len(args) != 2:
        return None

    key_type, value_type = args
    if key_type is not str:
        return None

    if dataclasses.is_dataclass(value_type):
        return value_type

    return None


def dc_to_dict(cls, expand_optional: bool = True) -> dict:
    """
    Convert a dataclass *class* to a plain dict using field defaults.

    When expand_optional=True, fields typed `SomeDataclass | None` that
    default to None are expanded to show the nested structure instead of null.
    """
    hints = typing.get_type_hints(cls)
    result = {}
    for f in dataclasses.fields(cls):
        val = _field_default(f)
        hint = hints.get(f.name)

        # Expand Optional[SomeDataclass] even when default is None
        if val is None and expand_optional and hint is not None:
            inner = _unwrap_optional_dataclass(hint)
            if inner is not None:
                result[f.name] = dc_to_dict(inner, expand_optional=True)
                continue

            list_inner = _unwrap_list_dataclass(hint)
            if list_inner is not None:
                result[f.name] = [dc_to_dict(list_inner, expand_optional=True)]
                continue

        if isinstance(val, dict) and not val and expand_optional and hint is not None:
            dict_inner = _unwrap_dict_dataclass(hint)
            if dict_inner is not None:
                result[f.name] = {"<policy_name>": dc_to_dict(dict_inner, expand_optional=True)}
                continue

        if val == "???":
            result[f.name] = "???"
        elif dataclasses.is_dataclass(val):
            result[f.name] = dc_to_dict(type(val), expand_optional=expand_optional)
        elif isinstance(val, list):
            result[f.name] = [
                dc_to_dict(type(item), expand_optional=expand_optional)
                if dataclasses.is_dataclass(item)
                else item
                for item in val
            ]
        elif isinstance(val, dict):
            result[f.name] = {
                key: dc_to_dict(type(item), expand_optional=expand_optional)
                if dataclasses.is_dataclass(item)
                else item
                for key, item in val.items()
            }
        elif isinstance(val, Enum):
            result[f.name] = val.value
        elif isinstance(val, Path):
            result[f.name] = str(val)
        else:
            result[f.name] = val

    return result


# ---------------------------------------------------------------------------
# YAML formatting helpers
# ---------------------------------------------------------------------------

def _yaml_dump(data) -> str:
    return yaml.safe_dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    ).rstrip()



# ---------------------------------------------------------------------------
# Build each section
# ---------------------------------------------------------------------------

def build_topics_section() -> dict:
    """One annotated example topic entry showing all optional sub-sections."""
    return {"<topic_key>": dc_to_dict(TopicConfig, expand_optional=True)}


def build_global_pre_processor_section() -> dict:
    return dc_to_dict(ImagePreprocessorConfig)


def build_synchronizer_section() -> dict:
    return dc_to_dict(SynchSubscriberNodeConfig)


def build_recorder_section() -> dict:
    return dc_to_dict(RecorderConfig)


def build_inference_section() -> dict:
    return dc_to_dict(InferenceConfig)


def build_data_conversion_section() -> dict:
    return dc_to_dict(DataConversionConfig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SECTIONS = [
    ("topics",                     ["recorder", "lerobot_eval_ros2", "data_conversion"], build_topics_section,
     "Add as many named topic entries as needed (e.g. cam0, cam1, joint_state, ...).\n"
     "  recorder, inference, and pre_processor are all optional per topic:\n"
     "  - recorder:     set this to include the topic in recorder.py data collection\n"
     "  - inference:    set this to include the topic as a policy observation in lerobot_eval_ros2.py\n"
     "  - pre_processor: per-topic image crop/resize override (falls back to global_image_pre_processor)"),
    ("global_image_pre_processor", ["lerobot_eval_ros2", "data_conversion"],             build_global_pre_processor_section,
     "Applied to image topics that have no per-topic pre_processor."),
    ("synchronizer",               ["recorder", "lerobot_eval_ros2"],                   build_synchronizer_section, ""),
    ("recorder",                   ["recorder"],                                         build_recorder_section,     ""),
    ("inference",                  ["lerobot_eval_ros2"],                               build_inference_section,    ""),
    ("data_conversion",            ["data_conversion"],                                  build_data_conversion_section, ""),
]


def _section(used_by: list[str], key: str, value, note: str = "") -> str:
    """Render one top-level YAML section with a comment header."""
    bar = "=" * 62
    used = " | ".join(used_by)
    lines = [f"# {bar}", f"# {key}   ({used})"]
    if note:
        for note_line in note.splitlines():
            lines.append(f"# {note_line}")
    lines.append(f"# {bar}")
    yaml_str = _inject_comments(_yaml_dump({key: value}), _ALL_FIELD_COMMENTS)
    lines.append(yaml_str)
    return "\n".join(lines) + "\n"


def generate() -> str:
    lines = [
        "# Auto-generated by scripts/generate_config.py — do not edit by hand.",
        "# Full config options and type annotations: medcvr_il/common/config_schema.py",
        "#   TopicConfig, ImagePreprocessorConfig, SynchSubscriberNodeConfig,",
        "#   RecorderConfig, InferenceConfig, DataConversionConfig",
        "# '???' marks required fields with no default.",
        "",
    ]
    for key, used_by, builder, note in SECTIONS:
        lines.append(_section(used_by, key, builder(), note))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", type=Path, default=Path("config/example_autogenerated.yaml"),
                        help="Output path (default: config/example_autogenerated.yaml)")
    args = parser.parse_args()

    content = generate()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
