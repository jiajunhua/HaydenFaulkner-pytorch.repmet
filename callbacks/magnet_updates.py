from utils.model_forward import forward


class UpdateClusters(object):
    def __init__(self, every, dataset):
        self.every = every
        self.dataset = dataset

    def __call__(self, epoch, batch, step, model, dataloaders, losses, optimizer, data, stats):
        if step % self.every == 0:
            print('Updating Clusters')
            outputs = forward(model=model,
                              dataset=self.dataset,
                              chunk_size=400)

            dataloaders['train'].batch_sampler.update_clusters(outputs)


class UpdateLosses(object):
    def __init__(self, every):
        self.every = every

    def __call__(self, epoch, batch, step, model, dataloaders, losses, optimizer, data, stats):
        if step % self.every == 0:
            dataloaders['train'].batch_sampler.update_losses(stats['sample_losses'])

