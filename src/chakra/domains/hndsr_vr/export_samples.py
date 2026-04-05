"""Export the fixed sample strips for a completed run."""

from __future__ import annotations

import argparse

from chakra.core.utils import get_device, load_config, prepare_workspace_temp, set_seed

from .evaluate_runner import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed sample comparisons")
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument("--run-name", required=True, help="Run name used for output folders")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path")
    parser.add_argument("--device", default=None, help="Optional device override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prepare_workspace_temp(config["paths"]["artifact_root"])
    set_seed(config["seed"])
    evaluate(config, args.run_name, get_device(args.device), args.checkpoint)


if __name__ == "__main__":
    main()
