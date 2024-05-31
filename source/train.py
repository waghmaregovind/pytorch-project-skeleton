"""
Training file
"""

from copy import deepcopy

import time
import logging
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from source import logger as log_utils
from source import utils
from source.models import AutoEncoder


class Trainer():
    """
    Trainer class
    """

    def __init__(self, args: argparse.Namespace,
                 model: nn.Module, optim: torch.optim, train_dl: DataLoader,
                 val_dl: DataLoader, criterion: torch.nn.modules.loss,
                 logger: logging.RootLogger, device: torch.device):
        """
        Training setting

        Args:
            args: arguments
            model: Model backbone
            optim: optimizer
            train_dl: train dataloader
            val_dl: validation dataloader
            criterion: loss function
            logger: logger
            device: torch device
        """
        self.args = args
        self.model = model
        self.optim = optim
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.criterion = criterion
        self.logger = logger
        self.device = device
        self.patience = 0
        self.best_loss = np.inf
        self.best_model = None

    def train_step(self) -> tuple[float, float]:
        """
        Training over one epoch

        Return 
            avg_epoch_loss: float, average epoch loss 
            tot_time: float, training time of the epoch
        """
        st_time = time.time()
        ep_reconstruction_loss = []
        self.model.train()

        for batch in tqdm(self.train_dl):
            batch = batch.to(self.device)
            embedding = self.model.encoder(x=batch)
            x_reconstructed = self.model.decoder(embedding=embedding)
            loss = self.criterion(x_reconstructed, batch)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            ep_reconstruction_loss.append(loss.item())

        avg_epoch_loss = np.mean(ep_reconstruction_loss)
        end_time = time.time()
        tot_time = (end_time - st_time) / 60
        return avg_epoch_loss, tot_time

    @torch.no_grad()
    def val_step(self) -> tuple[float, float]:
        """
        Validation over entire val dataset

        Return  
            avg_val_loss: float, validation loss for the dataset 
            tot_time: float, validation time
        """
        st_time = time.time()
        ep_reconstruction_loss = []
        self.model.eval()

        for x in tqdm(self.val_dl):
            x = x.to(self.device)
            embedding = self.model.encoder(x)
            x_reconstructed = self.model.decoder(embedding)
            loss = self.criterion(x_reconstructed, x)
            ep_reconstruction_loss.append(loss.item())

        avg_val_loss = np.mean(ep_reconstruction_loss)
        end_time = time.time()
        tot_time = (end_time - st_time) / 60
        return avg_val_loss, tot_time


    def train_epoch_end(self, val_loss: float):
        """
        Early stopping logic on training epoch end

        Args:
            param val_loss: float, validation loss at dataset level
        """

        if(self.best_loss - val_loss) < self.args.patience_threshold:
            self.patience += 1

            if self.best_loss > val_loss:
                self.best_loss = val_loss
                self.best_model = deepcopy(self.model.state_dict())

        else:
            self.best_loss = val_loss
            self.best_model = deepcopy(self.model.state_dict())
            self.patience = 0


    def fit(self):
        """
        Overall model training and validation
        """
        self.patience = 0
        self.best_loss = np.inf
        self.best_model = deepcopy(self.model.state_dict())

        for epoch in range(self.args.max_epochs):
            train_loss, train_time = self.train_step()
            val_loss, val_time = self.val_step()
            self.train_epoch_end(val_loss)

            if self.patience >= self.args.max_patience:
                self.logger.info(f'Breaking due to early stopping at epoch: {epoch}')

            fmt_str = (
                f'[E:{epoch:3d}, Train_Loss:{train_loss:.4f}, '
                f'Train_Time:{train_time:.4f}, Val_Loss:{val_loss:.4f}, '
                f'Val_Time:{val_time:.4f}, Patience:{self.patience}]'
            )

            self.logger.info(fmt_str)

        utils.save_model(self.args, self.best_model, self.logger)


def train_model(args: argparse.Namespace,
                logger: logging.RootLogger,
                device: torch.device):
    """
    Setup training pipeline

    Args:
        args: arguments
        logger: logger 
        device: torch device to use
    """

    # dataset
    train_data = utils.get_dataset(args=args, split='train')
    train_dl = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_data = utils.get_dataset(args=args, split='val')
    val_dl = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    # model
    model = AutoEncoder(num_input_feat=args.num_input_feat,
                        emb_dim=args.emb_dim)
    model = model.to(device)

    utils.log_model_details(args=args, model=model, logger=logger)

    # optimizer
    optim = utils.get_optimizer(args=args, model=model)

    # criterion
    criterion = utils.get_criterion()

    # model training object
    trainer = Trainer(
        args=args,
        model=model,
        optim=optim,
        train_dl=train_dl,
        val_dl=val_dl,
        criterion=criterion,
        logger=logger,
        device=device
    )

    trainer.fit()


def main(args: argparse.Namespace):
    """
    Driver method

    Args:
        args: arguments
    """

    # init dir
    args.log_dir = f'{args.root_dir}/logs'
    args.model_save_dir = f'{args.root_dir}/models'
    args.data_dir = f'{args.root_dir}/data'

    # init logger
    logger = log_utils.init_logger(args=args, split='train')
    utils.log_args(args=args, logger=logger)

    # init cuda device
    device = utils.get_cuda_device(logger=logger, device_id=args.device_id)

    # model training wrapper
    train_model(args=args, logger=logger, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # project config
    parser.add_argument('--root_dir', type=str, default='./..')
    parser.add_argument('--exp_name', type=str, default='exp_001')

    # dataset config

    # model config
    parser.add_argument('--model_name', type=str, default='AutoEncoder')
    parser.add_argument('--num_input_feat', type=int, default=128)
    parser.add_argument('--emb_dim', type=int, default=32)

    # training config
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--model_save_step', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--optim_wt_decay', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--max_patience', type=int, default=5)
    parser.add_argument('--patience_threshold', type=float, default=1e-05)
    parser.add_argument('--device_id', type=int, default=0)

    # logging config
    parser.add_argument('--log_step', type=int, default=1)

    arguments = parser.parse_args()
    main(args=arguments)
