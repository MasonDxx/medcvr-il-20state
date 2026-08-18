import argparse
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import tyro

import rclpy

from medcvr_il.synchronizer import SynchSubscriberNode, Recorder
from medcvr_il.common.config_schema import RecorderConfig, SynchSubscriberNodeConfig, TopicConfig


@dataclass
class RunRecorderConfig:
    """Combined configuration for both synchronizer and recorder."""
    synchronizer: SynchSubscriberNodeConfig = field(default_factory=SynchSubscriberNodeConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    topics: dict[str, TopicConfig] = field(default_factory=dict)


def load_config(yaml_path: str, remaining_args: list) -> RunRecorderConfig:
    """Load topics, synchronizer, and recorder from YAML and apply CLI overrides except for topics."""
    loaded_cfg = OmegaConf.load(yaml_path)
    recorder_cfg = OmegaConf.masked_copy(
        loaded_cfg,
        ["topics", "synchronizer", "recorder"],
    )
    yaml_defaults = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(RunRecorderConfig), recorder_cfg)
    )
    cli_overrides = tyro.cli(RunRecorderConfig, default=yaml_defaults, args=remaining_args)
    return RunRecorderConfig(
        topics=yaml_defaults.topics,
        synchronizer=cli_overrides.synchronizer,
        recorder=cli_overrides.recorder,
    )

def main(args=None):
    rclpy.init(args=args)
    # add_help=False so that --help is forwarded to tyro.cli, which knows
    # about all nested config fields and prints their attribute docstrings.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default=None, help="Path to the YAML configuration file")
    args, remaining_args = parser.parse_known_args()

    if args.config is None:
        # No YAML supplied — let tyro show --help (or error on missing fields).
        tyro.cli(RunRecorderConfig, args=remaining_args)
        return

    # Load both configs together using a single tyro.cli call
    run_recorder_config = load_config(args.config, remaining_args)
    topics_config = run_recorder_config.topics
    synchronizer_config = run_recorder_config.synchronizer
    recorder_config = run_recorder_config.recorder

    print("Loaded SynchronizerConfig:", synchronizer_config)
    print("Loaded RecorderConfig:", recorder_config)

    sync_subscriber_node = SynchSubscriberNode(topics_config, synchronizer_config)
    recorder = Recorder(recorder_config, sync_subscriber_node)

    sync_subscriber_node.join()
    recorder.join()

if __name__ == '__main__':
    main()
