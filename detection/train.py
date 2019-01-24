"""
training script for detection

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

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Detection Network')
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

    input_size = None
    mean = None
    std = None

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
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'],
                                                       sampler=samplers['train'],
                                                       batch_size=config.train.batch_size,
                                                       num_workers=1)
    if config.val.every > 0:
        dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'],
                                                         sampler=samplers['val'],
                                                         batch_size=config.train.batch_size, # todo change to val
                                                         num_workers=1)

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
    lr = config.train.learning_rate
    params = []
    # for key, value in dict(model.parameters()).items():
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (config.train.double_bias + 1), \
                            'weight_decay': config.train.bias_decay and config.train.weight_decay or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': config.train.weight_decay}]

    if config.train.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif config.train.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=config.train.momentum)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                   gamma=config.train.lr_scheduler_gamma,
                                                   step_size=config.train.lr_scheduler_step)

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

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

    # make variable
    from torch.autograd import Variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)


    since = time.time()

    train_loss = []
    train_rpn_loss_cls = []
    train_rpn_loss_box = []
    train_rcnn_loss_cls = []
    train_rcnn_loss_bbox = []
    train_rpn_acc = []
    train_rcnn_acc = []
    val_loss = []
    val_rpn_acc = []
    val_rcnn_acc = []

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
        print('Doing %d batches...' % len(dataloaders['train']))
        for data in dataloaders['train']:  # this gets a batch (or an episode)
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            # zero the parameter gradients
            model.zero_grad()

            # forward
            # Get model outputs and calculate loss

            # outputs = model(inputs)
            gt_rois, rois, rois_label, cls_pred, bbox_pred, rpn_scores, rpn_bboxs, rpn_cls_scores, rpn_bbox_preds, anchors =\
                model(im_data, im_info, gt_boxes, num_boxes)

            outputs = (gt_rois, rois, rois_label, cls_pred, bbox_pred, rpn_scores, rpn_bboxs, rpn_cls_scores, rpn_bbox_preds, anchors)
            loss, rpn_loss_cls, rpn_loss_box, rcnn_loss_cls, rcnn_loss_bbox, rpn_acc, rcnn_acc = \
                losses['train'](input=outputs, target=[gt_boxes, num_boxes, im_info])

            # backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistics
            train_loss.append(loss.item())
            train_rpn_loss_cls.append(rpn_loss_cls.mean().item())
            train_rpn_loss_box.append(rpn_loss_box.mean().item())
            train_rcnn_loss_cls.append(rcnn_loss_cls.mean().item())
            train_rcnn_loss_bbox.append(rcnn_loss_bbox.mean().item())
            train_rpn_acc.append(rpn_acc.item())
            train_rcnn_acc.append(rcnn_acc.item())

            for callback in callbacks['batch_end']:
                callback(epoch, batch, step, model, dataloaders, losses, optimizer,
                         data={'inputs': data, 'outputs': None, 'labels': None},
                         stats={'Training_Loss': train_loss[-1],
                                'Training_RPN_Acc': train_rpn_acc[-1],
                                'Training_RCNN_Acc': train_rcnn_acc[-1],
                                'Training_RPN_Class_Loss': train_rpn_loss_cls[-1],
                                'Training_RPN_Box_Loss': train_rpn_loss_box[-1],
                                'Training_RCNN_Class_Loss': train_rcnn_loss_cls[-1],
                                'Training_RCNN_Box_Loss': train_rcnn_loss_bbox[-1]})

            batch += 1
            step += 1

        avg_loss = np.mean(train_loss[-batch:])
        avg_rpn_acc = np.mean(train_rpn_acc[-batch:])
        avg_rcnn_acc = np.mean(train_rcnn_acc[-batch:])

        print(
            'Avg Train: Total Loss {:.4f}, RPN Class Loss {:.4f}, RPN Box Loss {:.4f}, RPN Acc: {:.4f}, RCNN Class Loss {:.4f}, RCNN Box Loss {:.4f}, RCNN Acc: {:.4f}'.format(
                avg_loss, np.mean(train_rpn_loss_cls), np.mean(train_rpn_loss_box), avg_rpn_acc,
                np.mean(train_rcnn_loss_cls), np.mean(train_rcnn_loss_bbox), avg_rcnn_acc))
        logger.info(
            'Avg Train: Total Loss {:.4f}, RPN Class Loss {:.4f}, RPN Box Loss {:.4f}, RPN Acc: {:.4f}, RCNN Class Loss {:.4f}, RCNN Box Loss {:.4f}, RCNN Acc: {:.4f}'.format(
                avg_loss, np.mean(train_rpn_loss_cls), np.mean(train_rpn_loss_box), avg_rpn_acc,
                np.mean(train_rcnn_loss_cls), np.mean(train_rcnn_loss_bbox), avg_rcnn_acc))

        if lr_scheduler:
            lr_scheduler.step()

        # Validation?
        if config.val.every > 0 and (epoch+1) % config.val.every == 0:

            for callback in callbacks['validation_start']:
                callback(epoch, 0, step, model, dataloaders, losses, optimizer,
                         data={},
                         stats={})

            # model.eval()
            v_batch = 0
            val_loss = []
            val_acc = []
            val_rpn_loss_cls = []
            val_rpn_loss_box = []
            val_rcnn_loss_cls = []
            val_rcnn_loss_bbox = []
            print("Validation with %d batches" % len(dataloaders['val']))
            for data in dataloaders['val']:
                # v_inputs = v_inputs.to(device)
                # v_labels = v_labels.to(device)

                im_data.data.resize_(data[0].size()).copy_(data[0])
                im_info.data.resize_(data[1].size()).copy_(data[1])
                gt_boxes.data.resize_(data[2].size()).copy_(data[2])
                num_boxes.data.resize_(data[3].size()).copy_(data[3])

                # print(gt_boxes)

                with torch.set_grad_enabled(False):  # disables grad calculation as dont need it so can save mem
                # Get model outputs and calculate loss

                    gt_rois, rois, rois_label, cls_pred, bbox_pred, rpn_scores, rpn_bboxs, rpn_cls_scores, rpn_bbox_preds, anchors = \
                        model(im_data, im_info, gt_boxes, num_boxes)

                    outputs = (
                    gt_rois, rois, rois_label, cls_pred, bbox_pred, rpn_scores, rpn_bboxs, rpn_cls_scores, rpn_bbox_preds,
                    anchors)
                    loss, rpn_loss_cls, rpn_loss_box, rcnn_loss_cls, rcnn_loss_bbox, rpn_acc, rcnn_acc = \
                        losses['val'](input=outputs, target=[gt_boxes, num_boxes, im_info])

                # statistics
                val_loss.append(loss.item())
                val_rpn_acc.append(rpn_acc.item())
                val_rcnn_acc.append(rcnn_acc.item())
                val_rpn_loss_cls.append(rpn_loss_cls.mean().item())
                val_rpn_loss_box.append(rpn_loss_box.mean().item())
                val_rcnn_loss_cls.append(rcnn_loss_cls.mean().item())
                val_rcnn_loss_bbox.append(rcnn_loss_bbox.mean().item())

                for callback in callbacks['validation_batch_end']:
                    callback(epoch, batch, step, model, dataloaders, losses, optimizer,  # todo should we make this v_batch?
                             data={'inputs': data, 'outputs': None, 'labels': None},
                             stats={'Validation_Loss': val_loss[-1],
                                    'Validation_RPN_Acc': val_rpn_acc[-1],
                                    'Validation_RCNN_Acc': val_rcnn_acc[-1],
                                    'Validation_RPN_Class_Loss': val_rpn_loss_cls[-1],
                                    'Validation_RPN_Box_Loss': val_rpn_loss_box[-1],
                                    'Validation_RCNN_Class_Loss': val_rcnn_loss_cls[-1],
                                    'Validation_RCNN_Box_Loss': val_rcnn_loss_bbox[-1]})

                v_batch += 1

            avg_v_loss = np.mean(val_loss)
            avg_v_rpn_acc = np.mean(val_rpn_acc)
            avg_v_rcnn_acc = np.mean(val_rcnn_acc)

            print('Avg Validation: Total Loss {:.4f}, RPN Class Loss {:.4f}, RPN Box Loss {:.4f}, RPN Acc: {:.4f}, RCNN Class Loss {:.4f}, RCNN Box Loss {:.4f}, RCNN Acc: {:.4f}'.format(
                avg_v_loss, np.mean(val_rpn_loss_cls), np.mean(val_rpn_loss_box), avg_v_rpn_acc, np.mean(val_rcnn_loss_cls), np.mean(val_rcnn_loss_bbox), avg_v_rcnn_acc))
            logger.info('Avg Validation: Total Loss {:.4f}, RPN Class Loss {:.4f}, RPN Box Loss {:.4f}, RPN Acc: {:.4f}, RCNN Class Loss {:.4f}, RCNN Box Loss {:.4f}, RCNN Acc: {:.4f}'.format(
                avg_v_loss, np.mean(val_rpn_loss_cls), np.mean(val_rpn_loss_box), avg_v_rpn_acc, np.mean(val_rcnn_loss_cls), np.mean(val_rcnn_loss_bbox), avg_v_rcnn_acc))

            # Best validation accuracy yet?
            if avg_v_rcnn_acc > best_acc:
                best_acc = avg_v_rcnn_acc
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
                         data={'inputs': None, 'outputs': None, 'labels': None},
                         stats={'Avg_Validation_Loss': avg_v_loss,
                                'Avg_Validation_RPN_Acc': avg_v_rpn_acc,
                                'Avg_Validation_RCNN_Acc': avg_v_rcnn_acc,
                                'Avg_Validation_RPN_Class_Loss': np.mean(val_rpn_loss_cls),
                                'Avg_Validation_RPN_Box_Loss': np.mean(val_rpn_loss_box),
                                'Avg_Validation_RCNN_Class_Loss': np.mean(val_rcnn_loss_cls),
                                'Avg_Validation_RCNN_Box_Loss': np.mean(val_rcnn_loss_bbox)})

        # End of epoch callbacks
        for callback in callbacks['epoch_end']:
            callback(epoch, batch, step, model, dataloaders, losses, optimizer,
                     data={'inputs': None, 'outputs': None, 'labels': None},
                     stats={'Avg_Training_Loss': avg_loss, 'Avg_Training_Acc': avg_rcnn_acc})

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
        best_acc = avg_rcnn_acc
        # best_state = copy.deepcopy(model.state_dict())
        best_state = model.state_dict()
        if hasattr(losses['train'], 'reps'):
            reps = losses['train'].get_reps()
        else:
            reps = None
        save_checkpoint(config, epoch, model, optimizer, best_acc, reps=reps, is_best=True)

    return model, best_state, best_acc, train_loss, avg_rcnn_acc, val_loss, val_acc


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
