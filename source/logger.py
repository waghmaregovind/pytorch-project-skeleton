"""
Logging file
"""

import logging
from pathlib import Path
from datetime import datetime
import argparse


def init_logger(args: argparse.Namespace, split: str):
    """
    Initialize logger object

    Args:
        args: arguments
        split: separate logging for train and test
    """

    # create logging folder
    path_prefix = f'{args.log_dir}/{args.exp_name}/{split}'
    Path(f'{path_prefix}').mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # file handler
    curr_time = datetime.now().strftime('%y_%m_%d_%H_%M')
    log_filename = f'{path_prefix}/{curr_time}.log'
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)

    # formatter
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s - %(funcName)s %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
