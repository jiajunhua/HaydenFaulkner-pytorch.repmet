"""
evaluation script for classification

"""

# Imports
from __future__ import print_function

import os
import time
import copy
import pprint
import argparse

import numpy as np

import torch
import torchvision
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter
from datetime import datetime

from config.config import config, update_config, check_config
from utils.logging.logger import initialize_logger
from utils.checkpointing import load_checkpoint

from model_definitions.initialize import initialize_model
from data_loading.initialize import initialize_dataset, initialize_sampler
from losses.initialize import initialize_loss
from callbacks.tensorboard import TensorBoard, EmbeddingGrapher

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Classification Network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()

    return args

args = parse_args()

def evaluate():

    # setup seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # Setup the paths and dirs
    save_path = os.path.join(config.model.root_dir, config.model.type, config.model.id, config.run_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert os.path.exists(save_path), '{} does not exist'.format(save_path)

    # Setup the logger
    logger = initialize_logger(save_path=save_path, run_id=config.run_id)
    pprint.pprint(config)
    logger.info('testing config:{}\n'.format(pprint.pformat(config)))

    #################### MODEL ########################
    # Load the model definition
    model, input_size, output_size, mean, std = initialize_model(config=config,
                                                                 model_name=config.model.type,
                                                                 model_id=config.model.id)
    model = model.to(device)

    # Use the GPU and Parallel it
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    cudnn.benchmark = True

    #################### DATA ########################
    # Load set and get test labels from datasets

    datasets = dict()
    datasets['test'] = initialize_dataset(config=config,
                                          dataset_name=config.dataset.name,
                                          dataset_id=config.dataset.id,
                                          split='test',
                                          input_size=input_size,
                                          mean=mean,
                                          std=std)

    samplers = dict()
    samplers['test'] = initialize_sampler(config=config,
                                          sampler_name=config.test.sampler,
                                          dataset=datasets['test'],
                                          split='test')

    dataloaders = dict()
    dataloaders['test'] = torch.utils.data.DataLoader(datasets['test'], batch_sampler=samplers['test'])

    #################### LOSSES + METRICS ######################
    # Setup losses
    losses = dict()
    losses['test'] = initialize_loss(config=config,
                                     loss_name=config.test.loss,
                                     split='test',
                                     n_classes=datasets['test'].n_categories)

    # Setup Metrics
    metrics = dict()

    ################### CALLBACKS #####################
    # Setup Callbacks
    callbacks = dict()
    tb_sw = SummaryWriter(log_dir=os.path.join(save_path, 'tb'), comment=config.run_id)

    callbacks['epoch_end'] = []
    callbacks['batch_end'] = [EmbeddingGrapher(every=config.vis.test_plot_embed_every, tb_sw=tb_sw, tag='test', label_image=True)]


    # Load model params
    _, _, model, _, reps = load_checkpoint(config=config,
                                     resume_from=config.test.resume_from,
                                     model=model,
                                     optimizer=None)


    if hasattr(losses['test'], 'reps') and reps is not None:
        losses['test'].set_reps(reps)

    perform(config=config,
            model=model,
            dataloaders=dataloaders,
            losses=losses,
            metrics=metrics,
            callbacks=callbacks,
            is_inception=False)


def perform(config,
            model,
            dataloaders,
            losses,
            metrics,
            callbacks,
            is_inception=False):

    since = time.time()

    test_loss = []
    test_acc = []

    step = 0

    model.eval()
    # Iterate over data.
    batch = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Get model outputs and calculate loss
        outputs = model(inputs)
        loss, sample_losses, pred, acc = losses['test'](input=outputs, target=labels)

        # statistics
        test_loss.append(loss.item())
        test_acc.append(acc.item())

        for callback in callbacks['batch_end']:
            callback(0, batch, step, model, dataloaders, losses, None,
                     data={'inputs': inputs, 'outputs': outputs, 'labels': labels},
                     stats={'Testing Loss': test_loss[-1], 'Testing Acc': test_acc[-1]})

        batch += 1
        step += 1

    avg_loss = np.mean(test_loss)
    avg_acc = np.mean(test_acc)

    print('Avg Testing Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))

    for callback in callbacks['epoch_end']:
        callback(0, batch, step, model, dataloaders, losses, None,
                 data={'inputs': inputs, 'outputs': outputs, 'labels': labels},
                 stats={'Avg Testing Loss': avg_loss, 'Avg Testing Acc': avg_acc})


    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, test_loss, test_acc


def main():
    from utils.debug import set_working_dir
    # set the working directory as appropriate
    set_working_dir()

    print('Called with argument:', args)
    # update config
    update_config(args.cfg)
    check_config(config)
    evaluate()


if __name__ == '__main__':
    main()
