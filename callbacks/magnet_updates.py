
import torch

from utils.model_forward import forward


class UpdateClusters(object):
    def __init__(self, every, dataloader, dataset, batch_size=64):
        self.every = every
        self.dataloader = dataloader
        self.dataset = dataset
        self.batch_size = batch_size

    def __call__(self, epoch, batch, step, model, dataloaders, losses, optimizer, data, stats):
        if step % self.every == 0:
            print('Updating Clusters')
            outputs, labels = forward(model=model,
                                      dataset=self.dataset,
                                      batch_size=self.batch_size)

            # todo check these labels match those in the batch sampler

            self.dataloader.batch_sampler.update_clusters(outputs)


class UpdateLosses(object):
    def __init__(self, every, dataloader):
        self.every = every
        self.dataloader = dataloader

    def __call__(self, epoch, batch, step, model, dataloaders, losses, optimizer, data, stats):
        if step % self.every == 0:
            self.dataloader.batch_sampler.update_losses(stats['sample_losses'])


class SetClusterMeans(object):
    def __init__(self, every, eval_loss, dataloader):
        self.every = every
        self.eval_loss = eval_loss
        self.dataloader = dataloader

    # def __call__(self, *args, **kwargs):
    def __call__(self, epoch, batch, step, model, dataloaders, losses, optimizer, data, stats):
        if step % self.every == 0:
            # ensure performed after an UpdateClusters() callback
            self.eval_loss.cluster_means = self.dataloader.batch_sampler.centroids
            self.eval_loss.cluster_classes = self.dataloader.batch_sampler.cluster_classes


class SetEvalVariance(object):
    def __init__(self, every, eval_loss, training_loss):
        self.every = every
        self.eval_loss = eval_loss
        self.training_loss = training_loss

    def __call__(self, epoch, batch, step, model, dataloaders, losses, optimizer, data, stats):
        if step % self.every == 0:
            self.eval_loss.variance = torch.mean(self.training_loss.variances)
