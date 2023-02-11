import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

import os
import sys
import logging

from src.argument import Arguments


def get_parser():
    parser = Arguments()
    args = parser.parse()

    # select GPU if available
    use_cuda = torch.cuda.is_available()
    args.device = 'cuda:0' if use_cuda else 'cpu'

    return args

def get_logger(is_main=True, filename=None): 
    """
    Get logger to store information about the script and track events that occur

    :param
        is_main: bool
                true if training 
        filename: str
                name of log file
    
    :return
        loggger: logging
                log infomation
    """
    logger = logging.getLogger(__name__)

    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None: 
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S", 
        level=logging.INFO if is_main else logging.WARN, 
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", 
        handlers=handlers
    )       
    return logger


def get_gpu_utilization(logger):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    if logger == None:
        print(f"GPU memory occupied: {info.used//1024**2} MB.")
    else:
        logger.info(f"GPU memory occupied: {info.used//1024**2} MB.")

def init_checkpoint_folder(args):
    """
    Init checkpoint path and make checkpoint directory

    :param
        args: Arguments
                contains hyper-parameter
    """
    args.checkpoint_path = os.path.join(args.checkpoint_dir, args.name)
    args.checkpoint_exists = os.path.exists(args.checkpoint_path)
    if not args.checkpoint_exists:
        os.makedirs(args.checkpoint_path)

