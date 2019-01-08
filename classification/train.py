"""
training script for classification

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

from config.config import config, update_config, check_config
from utils.logging.logger import initialize_logger
from utils.checkpointing import save_checkpoint, load_checkpoint

from model_definitions.initialize import initialize_model
from data_loading.initialize import initialize_dataset, initialize_sampler
from losses.initialize import initialize_loss
from callbacks.initialize import initialize_callbacks


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Classification Network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()

    return args


def train():

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
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    #################### MODEL ########################
    # Load the model definition
    model, input_size, output_size, mean, std = initialize_model(config=config,
                                                                 model_name=config.model.type,
                                                                 model_id=config.model.id)
    model = model.to(device)

    # Load model params

    # Use the GPU and Parallel it
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    cudnn.benchmark = True

    #################### DATA ########################
    # Load set and get train and test labels from datasets

    datasets = dict()
    datasets['train'] = initialize_dataset(config=config,
                                           dataset_name=config.dataset.name,
                                           dataset_id=config.dataset.id,
                                           split='train',
                                           input_size=input_size,
                                           mean=mean,
                                           std=std)
    if config.val.every > 0:
        datasets['val'] = initialize_dataset(config=config,
                                             dataset_name=config.dataset.name,
                                             dataset_id=config.dataset.id,
                                             split='val',
                                             input_size=input_size,
                                             mean=mean,
                                             std=std)

    samplers = dict()
    samplers['train'] = initialize_sampler(config=config,
                                           sampler_name=config.train.sampler,
                                           dataset=datasets['train'],
                                           split='train')
    if config.val.every > 0:
        samplers['val'] = initialize_sampler(config=config,
                                             sampler_name=config.val.sampler,
                                             dataset=datasets['val'],
                                             split='val')

    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_sampler=samplers['train'])
    if config.val.every > 0:
        dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_sampler=samplers['val'])

    #################### LOSSES + METRICS ######################
    # Setup losses
    losses = dict()
    losses['train'] = initialize_loss(config=config,
                                      loss_name=config.train.loss,
                                      split='train',
                                      n_classes=datasets['train'].n_categories)
    if config.val.every > 0:
        losses['val'] = initialize_loss(config=config,
                                        loss_name=config.val.loss,
                                        split='val',
                                        n_classes=datasets['val'].n_categories)

    # Setup Optimizer
    optimizer = torch.optim.Adam(params=(list(filter(lambda p: p.requires_grad, model.parameters())) + list(losses['train'].parameters())),
                                 lr=config.train.learning_rate)

    if config.run_type == 'protonets':  # TODO consider putting in a callback on epoch_end, but then need to pass lr_sch
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                   gamma=config.train.lr_scheduler_gamma,
                                                   step_size=config.train.lr_scheduler_step)
    else:
        lr_scheduler = None

    ################### CALLBACKS #####################
    # Setup Callbacks
    callbacks = initialize_callbacks(config=config,
                                     model=model,
                                     datasets=datasets,
                                     samplers=samplers,
                                     dataloaders=dataloaders,
                                     losses=losses,
                                     optimizer=optimizer)

    # pre fit() flag
    if config.model.type == 'inception':
        is_inception = True
    else:
        is_inception = False

    fit(config=config,
        logger=logger,
        model=model,
        dataloaders=dataloaders,
        losses=losses,
        optimizer=optimizer,
        callbacks=callbacks,
        lr_scheduler=lr_scheduler,
        is_inception=is_inception)


def fit(config,
        logger,
        model,
        dataloaders,
        losses,
        optimizer,
        callbacks,
        lr_scheduler=None,
        is_inception=False,
        resume_from='L'):

    since = time.time()

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    best_state = copy.deepcopy(model.state_dict())
    best_state = model.state_dict()

    # Load Checkpoint?
    start_epoch, best_acc, model, optimizer, reps = load_checkpoint(config=config,
                                                                    resume_from=resume_from,
                                                                    model=model,
                                                                    optimizer=optimizer)

    if hasattr(losses['train'], 'reps') and reps is not None:
        losses['train'].set_reps(reps)

    for callback in callbacks['training_start']:
        callback(0, 0, 0, model, dataloaders, losses, optimizer,
                 data={},
                 stats={})

    step = start_epoch*len(dataloaders['train'])
    for epoch in range(start_epoch, config.train.epochs):
        print('Epoch {}/{}'.format(epoch, config.train.epochs - 1))
        logger.info('Epoch {}/{}'.format(epoch, config.train.epochs - 1))
        print('-' * 10)
        logger.info('-' * 10)

        for callback in callbacks['epoch_start']:
            callback(epoch, 0, step, model, dataloaders, losses, optimizer,
                     data={},
                     stats={})

        # Iterate over data.
        model.train()
        batch = 0
        for inputs, labels in dataloaders['train']:  # this gets a batch (or an episode)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            model.zero_grad()

            # forward
            # Get model outputs and calculate loss
            if is_inception:
                # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                outputs, aux_outputs = model(inputs)
                loss_main, sample_losses_main, pred, acc = losses['train'](input=outputs, target=labels)
                loss_aux, sample_losses_aux, pred_aux, acc_aux = losses['train'](input=aux_outputs, target=labels)
                loss = loss_main + 0.4 * loss_aux
                sample_losses = sample_losses_main + 0.4 * sample_losses_aux
            else:
                outputs = model(inputs)
                loss, sample_losses, pred, acc = losses['train'](input=outputs, target=labels)

            # backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            train_loss.append(loss.item())
            train_acc.append(acc.item())

            for callback in callbacks['batch_end']:
                callback(epoch, batch, step, model, dataloaders, losses, optimizer,
                         data={'inputs': inputs, 'outputs': outputs, 'labels': labels},
                         stats={'Training_Loss': train_loss[-1], 'Training_Acc': train_acc[-1],
                                'sample_losses': sample_losses})

            batch += 1
            step += 1

        avg_loss = np.mean(train_loss[-batch:])
        avg_acc = np.mean(train_acc[-batch:])

        print('Avg Training Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
        logger.info('Avg Training Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
        if lr_scheduler:
            lr_scheduler.step()

        # Validation?
        if config.val.every > 0 and (epoch+1) % config.val.every == 0:

            for callback in callbacks['validation_start']:
                callback(epoch, 0, step, model, dataloaders, losses, optimizer,
                         data={},
                         stats={})

            model.eval()
            v_batch = 0
            val_loss = []
            val_acc = []
            for v_inputs, v_labels in dataloaders['val']:
                v_inputs = v_inputs.to(device)
                v_labels = v_labels.to(device)

                with torch.set_grad_enabled(False):  # disables grad calculation as dont need it so can save mem
                # Get model outputs and calculate loss
                    v_outputs = model(v_inputs)
                loss, sample_losses, pred, acc = losses['val'](input=v_outputs, target=v_labels)

                # statistics
                val_loss.append(loss.item())
                val_acc.append(acc.item())

                for callback in callbacks['validation_batch_end']:
                    callback(epoch, batch, step, model, dataloaders, losses, optimizer,  # todo should we make this v_batch?
                             data={'inputs': v_inputs, 'outputs': v_outputs, 'labels': v_labels},
                             stats={'Validation_Loss': val_loss[-1], 'Validation_Acc': val_acc[-1]})

                v_batch += 1

            avg_v_loss = np.mean(val_loss)
            avg_v_acc = np.mean(val_acc)

            print('Avg Validation Loss: {:.4f} Acc: {:.4f}'.format(avg_v_loss, avg_v_acc))
            logger.info('Avg Validation Loss: {:.4f} Acc: {:.4f}'.format(avg_v_loss, avg_v_acc))

            # Best validation accuracy yet?
            if avg_v_acc > best_acc:
                best_acc = avg_v_acc
                # best_state = copy.deepcopy(model.state_dict())
                best_state = model.state_dict()
                if hasattr(losses['train'], 'reps'):
                    reps = losses['train'].get_reps()
                else:
                    reps = None
                save_checkpoint(config, epoch, model, optimizer, best_acc, reps=reps, is_best=True)

            # End of validation callbacks
            for callback in callbacks['validation_end']:
                callback(epoch, batch, step, model, dataloaders, losses, optimizer,
                         data={'inputs': v_inputs, 'outputs': v_outputs, 'labels': v_labels},
                         stats={'Avg_Validation_Loss': avg_v_loss, 'Avg_Validation_Acc': avg_v_acc})

        # End of epoch callbacks
        for callback in callbacks['epoch_end']:
            callback(epoch, batch, step, model, dataloaders, losses, optimizer,
                     data={'inputs': inputs, 'outputs': outputs, 'labels': labels},
                     stats={'Avg_Training_Loss': avg_loss, 'Avg_Training_Acc': avg_acc})

        # Checkpoint?
        if config.train.checkpoint_every > 0 and epoch % config.train.checkpoint_every == 0:
            if hasattr(losses['train'], 'reps'):
                reps = losses['train'].get_reps()
            else:
                reps = None
            save_checkpoint(config, epoch, model, optimizer, best_acc, reps=reps, is_best=False)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
    logger.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    logger.info('Best val Acc: {:4f}'.format(best_acc))

    for callback in callbacks['training_end']:
        callback(epoch, batch, step, model, dataloaders, losses, optimizer,
                 data={},
                 stats={})

    # If no validation we save the last model as best
    if config.val.every < 1:
        best_acc = avg_acc
        # best_state = copy.deepcopy(model.state_dict())
        best_state = model.state_dict()
        if hasattr(losses['train'], 'reps'):
            reps = losses['train'].get_reps()
        else:
            reps = None
        save_checkpoint(config, epoch, model, optimizer, best_acc, reps=reps, is_best=True)

    return model, best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def main():
    from utils.debug import set_working_dir
    # set the working directory as appropriate
    set_working_dir()

    args = parse_args()

    print('Called with argument:', args)
    # update config
    update_config(args.cfg)
    check_config(config)
    train()


if __name__ == '__main__':
    main()
