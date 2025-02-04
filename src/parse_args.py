import argparse
from dataclasses import dataclass


@dataclass
class Args:
    data_prefix: str
    config_path: str


def parse_args(args: list[str]) -> Args:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Main file to run experiments for the reproduction of the GraphSAINT paper",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        dest="config_path",
        help="Path to .yaml config file",
    )
    parser.add_argument(
        "-d",
        "--data_prefix",
        required=True,
        type=str,
        help="Dataset prefix (e.g. 'reddit')",
    )
    args = parser.parse_args()
    return Args(data_prefix=args.data_prefix, config_path=args.config_path)
