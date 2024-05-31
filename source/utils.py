"""
Utils file
"""

import logging
from pathlib import Path
import argparse

import torch
from torch import nn


def get_cuda_device(logger: logging.RootLogger, device_id: int) -> torch.device:
    """
    GPU setting

    Args:
        logger: logging.RootLogger
        device_id: int, GPU device id 

    Returns:
        device: torch.device, torch device 
    """

    if torch.cuda.is_available():
        torch.cuda.set_device(device=device_id)
        logger.info(f'torch.__verspion__: {torch.__version__}')
        logger.info(f'torch.cuda.device_count: {torch.cuda.device_count()}')
        logger.info(f'torch.cuda.current_device: {torch.cuda.current_device()}')
        logger.info(f'torch.cuda.mem_get_info: {torch.cuda.mem_get_info()}')
    else:
        logger.info('CUDA NOT AVAILABLE')

    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    return device


def log_model_details(args: argparse.Namespace, model: nn.Module, logger: logging.RootLogger):
    """
    Log model details

    Args:
        args: arguments
        model: model under consideration
        logger: logger object
    """
    logger.info('-'*100)
    logger.info(f'Model name: {args.model_name}')
    logger.info(f'Model: {model}')
    param_size = sum([p.numel() for p in model.parameters() if p.requires_grad])
    param_size_bytes = param_size * 4
    param_size_kb = param_size * 4 /1024
    param_size_mb = param_size * 4 / (1024 * 1024)
    logger.info(f'Model size: {param_size_bytes} Bytes')
    logger.info(f'Model size: {param_size_kb} KB')
    logger.info(f'Model size: {param_size_mb} MB')
    logger.info('-'*100)


def get_dataset(args: argparse.Namespace, split: str) -> torch.Tensor:
    """
    Utility to load dataset. Right now data is generated randomly for demo.

    Args:
        args: arguments
        split: train, val or test split 
    """

    # train, val and test file paths
    if split == 'train':
        path = f'{args.data_dir}/train.npy'
    elif split == 'val':
        path = f'{args.data_dir}/val.npy'
    elif split == 'test':
        path = f'{args.data_dir}/test.npy'

    # dataset = np.load(path)
    # random data generation for demo
    dataset = torch.rand(4096, 128)
    return dataset


def log_args(args: argparse.Namespace, logger: logging.RootLogger):
    """
    Logging arguments for the experiment
    """
    logger.info('-'*100)
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')
    logger.info('-'*100)


def get_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim:
    """
    Initialize the optimizer
    """

    train_params = model.parameters()
    optim = torch.optim.Adam(train_params, weight_decay=args.optim_wt_decay, lr=args.lr)
    return optim


def get_criterion() -> torch.nn.modules.loss:
    """
    Define loss function
    """
    loss_function = torch.nn.MSELoss()
    return loss_function


def save_model(args: argparse.Namespace, model: nn.Module, logger: logging.RootLogger):
    """
    Utility to save model
    """
    model_savepath = f'{args.model_save_dir}/{args.model_name}'
    Path(f'{model_savepath}').mkdir(parents=True, exist_ok=True)
    model_savepath = f'{model_savepath}/{args.exp_name}.pt'
    logger.info(f'Saving model at {model_savepath}')
    torch.save(model, model_savepath)
