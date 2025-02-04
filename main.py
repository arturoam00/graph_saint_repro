#!/usr/bin/env python3

import datetime as dt
import logging
import os
import sys
import time

import yaml

from src.data_loader import DataLoader
from src.parse_args import Args, parse_args
from src.train import train

LOG_DIR = os.path.abspath("./log_files")


def main(args: Args) -> None:
    ###### Set up logging ######

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    timestamp = dt.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(
        LOG_DIR,
        f"{os.path.basename(args.config_path).replace(".yaml", "")}-{timestamp}.log",
    )
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(filename=log_path, mode="w"),
        ],
    )
    logger = logging.getLogger(f"{__name__}")

    def log_exceptions(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(
                exc_type, exc_value, exc_traceback
            )  # Don't log Ctrl+C exit
            return
        logger.error(
            "Uncaught Exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    # Install exception handler
    sys.excepthook = log_exceptions

    ###### Load config ######

    logger.info(f" -- Loading config from '{args.config_path}' --")

    with open(args.config_path, "r") as cf:
        config = yaml.safe_load(cf)

    logger.info(
        f"Config loaded: \n\t{"\n\t".join(f"{k}: {v}" for k, v in config.items())}"
    )

    ###### Load data ######

    dl = DataLoader(prefix=args.data_prefix)
    config.update({"n_classes": dl.n_classes})

    train(data=dl.get_data(), **config)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
