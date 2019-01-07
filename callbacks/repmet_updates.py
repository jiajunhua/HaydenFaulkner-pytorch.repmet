from utils.model_forward import forward
from sklearn.cluster import KMeans


class UpdateReps(object):
    def __init__(self, every, dataset, batch_size=64):
        self.every = every
        self.dataset = dataset
        self.batch_size = batch_size

    def __call__(self, epoch, batch, step, model, dataloaders, losses, optimizer, data, stats):
        if step % self.every == 0:
            print('Updating Reps')
            outputs, labels = forward(model=model,
                                      dataset=self.dataset,
                                      batch_size=self.batch_size)

            N = losses['train'].N  # todo might be a nicer way to get these
            k = losses['train'].k

            for c in range(N):
                class_mask = labels == c
                class_examples = outputs[class_mask]
                kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=20)
                kmeans.fit(class_examples)

                start = c * k
                stop = (c+1) * k
                # losses['train'].reps.data[start:stop] = torch.Tensor(kmeans.cluster_centers_).cuda().float()
                losses['train'].set_reps(kmeans.cluster_centers_, start, stop)


class UpdateValReps(object):

    def __init__(self, every):
        self.every = every

    def __call__(self, epoch, batch, step, model, dataloaders, losses, optimizer, data, stats):
        if step % self.every == 0:
            losses['val'].reps = losses['train'].reps