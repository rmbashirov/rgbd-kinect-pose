import os
import collections
import pydoc

import hydra
from omegaconf import OmegaConf

import torch
from torch import nn, optim

import face_expression
from face_expression import utils


@hydra.main(config_path="face_expression/config", config_name="config.yaml")
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True))
    print()

    # setup environment
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    utils.common.setup_environment(config.random_seed)

    # setup distributed
    # config.world_size = os.environ.get('WORLD_SIZE', 1)
    # config.distributed = config.world_size > 1

    # if config.distributed:
    #     torch.cuda.set_device(config.local_rank)
    #     torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # get dataloaders
    dataloaders = utils.misc.get_dataloaders(config, splits=('train', 'val'))
    print(">>> Successfully loaded dataloaders")
    
    # load runner
    runner_cls = pydoc.locate(config.runner.cls)
    runner = runner_cls(config)
    runner = runner.to(config.device)
    print(">>> Successfully loaded runner")

    # optimizer
    optimizer = runner.get_optimizer()
    print(">>> Successfully loaded optimizer")

    # setup experiment
    logger = utils.misc.get_logger(config)
    print(">>> Successfully loaded logger")

    # train loop
    for epoch in range(config.loop.n_epochs):
        runner.state['epoch'] = epoch

        if config.local_rank == 0:
            print("Epoch {}/{}".format(runner.state['epoch'], config.loop.n_epochs))

        runner.run_epoch(dataloaders['train'], optimizer, logger, 'train', n_iters_per_epoch=config.data.train.n_iters_per_epoch)
        runner.run_epoch(dataloaders['val'], None, logger, 'val', n_iters_per_epoch=config.data.val.n_iters_per_epoch)


if __name__ == '__main__':
    main()
