"""
Testing file
"""

import time
import logging
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm.auto import tqdm


from source import logger as log_utils
from source import utils
from source.models import AutoEncoder


class Tester():
    """
    Tester class
    """

    def __init__(self, args: argparse.Namespace,
                 model: nn.Module, test_dl: DataLoader,
                 criterion: torch.nn.modules.loss,
                 logger: logging.RootLogger, device: torch.device):
        """
        Testing setting

        Args:
            args: arguments
            model: Model backbone
            test_dl: test dataloader
            criterion: loss function
            logger: logger
            device: torch device
        """

        self.args = args
        self.model = model
        self.test_dl = test_dl
        self.criterion = criterion
        self.logger = logger
        self.device = device

    @torch.no_grad()
    def test_step(self) -> tuple[float, float]:
        """
        Test over entire val dataset

        Return  
            avg_test_loss: float, test loss for the dataset 
            tot_time: float, test time
        """
        st_time = time.time()
        ep_reconstruction_loss = []
        self.model.eval()

        for x in tqdm(self.test_dl):
            x = x.to(self.device)
            embedding = self.model.encoder(x=x)
            x_reconstructed = self.model.decoder(embedding=embedding)
            loss = self.criterion(x_reconstructed, x)
            ep_reconstruction_loss.append(loss.item())

        avg_test_loss = np.mean(ep_reconstruction_loss)
        end_time = time.time()
        tot_time = (end_time - st_time) / 60
        return avg_test_loss, tot_time

    def fit(self):
        """
        Overall model testing
        """
        model_savepath = (
            f'{self.args.model_save_dir}/'
            f'{self.args.model_name}/'
            f'{self.args.exp_name}.pt'
        )
        self.logger.info(f'Loading model from {model_savepath}')
        self.model.load_state_dict(torch.load(model_savepath))
        test_loss, test_time = self.test_step()
        fmt_str = f'[Test_Loss:{test_loss:.4f}, Test_Time:{test_time:.4f}]'
        self.logger.info(fmt_str)


def test_model(args: argparse.Namespace, logger: logging.RootLogger, device: torch.device):
    """
    Setup test pipeline

    Args:
        args: arguments
        logger: logger 
        device: torch device to use
    """

    # dataset
    test_data = utils.get_dataset(args=args, split='test')
    test_dl = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    # model
    model = AutoEncoder(num_input_feat=args.num_input_feat, emb_dim=args.emb_dim)
    model = model.to(device)

    utils.log_model_details(args=args, model=model, logger=logger)

    # criterion
    criterion = utils.get_criterion()

    # model testing object
    tester = Tester(
        args=args, model=model,
        test_dl=test_dl,
        criterion=criterion,
        logger=logger,
        device=device
    )
    tester.fit()


def main(args):
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
    logger = log_utils.init_logger(args=args, split='test')
    utils.log_args(args, logger)

    # init cuda device
    device = utils.get_cuda_device(logger=logger, device_id=args.device_id)

    # model testing wrapper
    test_model(args=args, logger=logger, device=device)


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

    # test config
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device_id', type=int, default=0)

    arguments = parser.parse_args()
    main(args=arguments)


# python test.py --exp_name exp_001
