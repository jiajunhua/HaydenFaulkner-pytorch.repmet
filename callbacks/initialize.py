import os
import warnings

from tensorboardX import SummaryWriter

from callbacks.tensorboard import TensorBoard, EmbeddingGrapher
from callbacks.magnet_updates import UpdateClusters, UpdateLosses, SetClusterMeans, SetEvalVariance
from callbacks.repmet_updates import UpdateReps, UpdateValReps


def initialize_callbacks(config, model, datasets, samplers, dataloaders, losses, optimizer):

    callbacks = {'training_start': [],
                 'epoch_start': [],
                 'batch_start': [],
                 'batch_end': [],
                 'validation_start': [],
                 'validation_batch_start': [],
                 'validation_batch_end': [],
                 'validation_end': [],
                 'epoch_end': [],
                 'training_end': []
                 }

    # init the tensorboard summary writer that we will write to with callbacks
    tb_sw = SummaryWriter(log_dir=os.path.join(config.model.root_dir, config.model.type, config.model.id, config.run_id, 'tb'), comment=config.run_id)

    if config.run_type == 'protonets':

        callbacks['batch_end'] = [TensorBoard(every=config.vis.every, tb_sw=tb_sw),
                                  EmbeddingGrapher(every=config.vis.plot_embed_every, tb_sw=tb_sw, tag='train', label_image=True)]

        callbacks['epoch_end'] = [TensorBoard(every=config.vis.every, tb_sw=tb_sw)]

        callbacks['validation_end'] = [TensorBoard(every=config.vis.every, tb_sw=tb_sw),
                                       EmbeddingGrapher(every=config.vis.plot_embed_every, tb_sw=tb_sw, tag='val',
                                                        label_image=True)]

    elif config.run_type == 'magnetloss':

        callbacks['epoch_start'] = [UpdateClusters(every=1, dataloader=dataloaders['train'], dataset=datasets['train'], batch_size=config.train.for_bs)]

        callbacks['batch_end'] = [TensorBoard(every=config.vis.every, tb_sw=tb_sw),
                                  EmbeddingGrapher(every=config.vis.plot_embed_every, tb_sw=tb_sw, tag='train', label_image=True),
                                  UpdateLosses(every=1, dataloader=dataloaders['train']),
                                  UpdateClusters(every=10, dataloader=dataloaders['train'], dataset=datasets['train'], batch_size=config.train.for_bs)]

        callbacks['epoch_end'] = [TensorBoard(every=config.vis.every, tb_sw=tb_sw)]

        # Update the validation clusters with training data and set them in the val loss with the variance from training
        # so we can perform the evaluation
        callbacks['validation_start'] = [UpdateClusters(every=1, dataloader=dataloaders['train'], dataset=datasets['train'], batch_size=config.train.for_bs),
                                         SetClusterMeans(every=1, eval_loss=losses['val'], dataloader=dataloaders['train']),
                                         SetEvalVariance(every=1, eval_loss=losses['val'], training_loss=losses['train'])]

        callbacks['validation_end'] = [TensorBoard(every=config.vis.every, tb_sw=tb_sw),
                                       EmbeddingGrapher(every=config.vis.plot_embed_every, tb_sw=tb_sw, tag='val', label_image=True)]

    elif config.run_type == 'repmet':
        callbacks['training_start'] = [UpdateReps(every=1, dataset=datasets['train'], batch_size=config.train.for_bs)]

        callbacks['batch_end'] = [TensorBoard(every=config.vis.every, tb_sw=tb_sw),
                                  EmbeddingGrapher(every=config.vis.plot_embed_every, tb_sw=tb_sw, tag='train', label_image=True)]

        callbacks['epoch_end'] = [TensorBoard(every=config.vis.every, tb_sw=tb_sw)]

        callbacks['validation_start'] = [UpdateValReps(every=1)]
        callbacks['validation_end'] = [TensorBoard(every=config.vis.every, tb_sw=tb_sw),
                                       EmbeddingGrapher(every=config.vis.plot_embed_every, tb_sw=tb_sw, tag='val',
                                                        label_image=True)]

    elif config.run_type == 'detection':

        callbacks['batch_end'] = [TensorBoard(every=config.vis.every, tb_sw=tb_sw)]

        callbacks['epoch_end'] = [TensorBoard(every=1, tb_sw=tb_sw)]

        callbacks['validation_end'] = [TensorBoard(every=1, tb_sw=tb_sw)]

    else:
        warnings.warn(config.run_type + "not recognised, no callbacks initialised.")

    return callbacks
