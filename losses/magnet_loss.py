"""
Taken from vithursant's repo:
https://github.com/vithursant/MagnetLoss-PyTorch/blob/master/magnet_loss/magnet_loss.py
"""
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class MagnetLoss(nn.Module):
    """
    Magnet loss technique presented in the paper:
    ''Metric Learning with Adaptive Density Discrimination'' by Oren Rippel, Manohar Paluri, Piotr Dollar, Lubomir Bourdev in
    https://research.fb.com/wp-content/uploads/2016/05/metric-learning-with-adaptive-density-discrimination.pdf?

    Args:
        r: A batch of features.
        classes: Class labels for each example.
        clusters: Cluster labels for each example.
        cluster_classes: Class label for each cluster.
        n_clusters: Total number of clusters.
        alpha: The cluster separation gap hyperparameter.

    Returns:
        total_loss: The total magnet loss for the batch.
        losses: The loss for each example in the batch.
    """
    def __init__(self, m, d, alpha=1.0, L=128, style='closest'):
        super(MagnetLoss, self).__init__()
        self.r = None
        self.classes = None
        self.clusters = None
        self.cluster_classes = None
        self.n_clusters = None
        self.alpha = alpha
        self.L = L
        self.style = style
        self.n_clusters = m
        self.examples_per_cluster = d
        self.variances = torch.tensor([0.0])

    def forward(self, input, target):  # reps and classes, x and y

        GPU_INT_DTYPE = torch.cuda.IntTensor
        GPU_LONG_DTYPE = torch.cuda.LongTensor
        GPU_FLOAT_DTYPE = torch.cuda.FloatTensor

        self.r = input
        classes = target.cpu().numpy()
        self.classes = torch.from_numpy(classes).type(GPU_LONG_DTYPE)
        self.clusters, _ = torch.sort(torch.arange(0, float(self.n_clusters)).repeat(self.examples_per_cluster))
        self.clusters = self.clusters.type(GPU_INT_DTYPE)
        self.cluster_classes = self.classes[0:self.n_clusters*self.examples_per_cluster:self.examples_per_cluster]

        # Take cluster means within the batch
        cluster_examples = dynamic_partition(self.r, self.clusters, self.n_clusters)

        cluster_means = torch.stack([torch.mean(x, dim=0) for x in cluster_examples])

        sample_costs = compute_euclidean_distance(cluster_means, expand_dims(self.r, 1))

        self.sample_costs = sample_costs

        clusters_tensor = self.clusters.type(GPU_FLOAT_DTYPE)
        n_clusters_tensor = torch.arange(0, self.n_clusters).type(GPU_FLOAT_DTYPE)

        intra_cluster_mask = Variable(comparison_mask(clusters_tensor, n_clusters_tensor).type(GPU_FLOAT_DTYPE))

        intra_cluster_costs = torch.sum(intra_cluster_mask * sample_costs, dim=1)

        N = self.r.size()[0]  # N = M*D (Batch size)

        variance = torch.sum(intra_cluster_costs) / float(N - 1)

        # self.variances = np.hstack((self.variances, variance.data.cpu().numpy()))
        self.variances = torch.cat((self.variances, variance.unsqueeze(0).cpu()), 0)

        var_normalizer = -1 / (2 * variance**2)

        # Compute numerator
        numerator = torch.exp(var_normalizer * intra_cluster_costs - self.alpha)

        classes_tensor = self.classes.type(GPU_FLOAT_DTYPE)
        cluster_classes_tensor = self.cluster_classes.type(GPU_FLOAT_DTYPE)

        # Compute denominator
        diff_class_mask = Variable(comparison_mask(classes_tensor, cluster_classes_tensor).type(GPU_FLOAT_DTYPE))

        diff_class_mask = 1 - diff_class_mask  # Logical not on ByteTensor

        denom_sample_costs = torch.exp(var_normalizer * sample_costs)

        denominator = torch.sum(diff_class_mask * denom_sample_costs, dim=1)

        epsilon = 1e-8

        losses = F.relu(-torch.log(numerator / (denominator + epsilon) + epsilon))

        total_loss = torch.mean(losses)


        if self.style == 'closest':  # acts on the clusters in this batch/episode rather than those calculate over the entire set!!
            _, pred = sample_costs.min(1)
            acc = pred.eq(clusters_tensor.type(GPU_LONG_DTYPE)).float().mean()
        else:
            raise NotImplementedError
            # TODO implement the version that takes into account variance
            # TODO note it will still just be acc on batch rather than set... (unlike val)
            # num_classes = len(np.unique(self.cluster_classes.cpu()))  # m # the number of classes in this batch
            #
            # num_clusters = cluster_means.size()[0]  # m*k
            #
            # # Sort the clusters by closest distance to sample
            # sorted_sample_costs, indices = torch.sort(sample_costs)
            # sorted_sample_costs = sorted_sample_costs.squeeze()
            # indices = indices.type(GPU_LONG_DTYPE).squeeze()
            # sorted_cluster_classes = self.cluster_classes[indices]
            #
            # # If L < num_clusters then lets only take the top L
            # if self.L < num_clusters:
            #     sorted_sample_costs = sorted_sample_costs[:self.L]
            #     sorted_cluster_classes = sorted_cluster_classes[:self.L]
            #     num_clusters = self.L
            #
            # normalised_costs = torch.exp(var_normalizer * sorted_sample_costs).type(GPU_FLOAT_DTYPE)
            #
            # per_class_costs = torch.zeros(num_classes, num_clusters).type(GPU_FLOAT_DTYPE)  # todo, address this issue of num_classes not matching batch_size and that being a problem...
            # per_class_costs = per_class_costs.scatter_(0, sorted_cluster_classes.unsqueeze(0), normalised_costs.unsqueeze(0))
            # numerator = per_class_costs.sum(1)
            #
            # denominator = torch.sum(normalised_costs)
            #
            # epsilon = 1e-8
            #
            # probs = numerator / (denominator + epsilon)
            #
            # _, pred = probs.max(0)
            # acc = pred.eq(target).float()
        return total_loss, losses, pred, acc


class MagnetLossEval(nn.Module):

    def __init__(self, L=128, style='magnet'):
        super(MagnetLossEval, self).__init__()
        self.cluster_means = None
        self.cluster_classes = None
        self.variance = None
        self.L = L
        self.style = style

    def forward(self, input, target):  # reps and classes, x and y # expects batch size of 1

        # make sure these have been set with the callbacks!!
        assert self.cluster_means is not None
        assert self.cluster_classes is not None
        assert self.variance is not None

        GPU_INT_DTYPE = torch.cuda.IntTensor
        GPU_LONG_DTYPE = torch.cuda.LongTensor
        GPU_FLOAT_DTYPE = torch.cuda.FloatTensor

        num_classes = np.max(self.cluster_classes) + 1  # the number of classes of the dataset
        cluster_means = torch.from_numpy(self.cluster_means).type(GPU_FLOAT_DTYPE)
        cluster_classes = torch.from_numpy(self.cluster_classes).type(GPU_LONG_DTYPE)
        sample_costs = compute_euclidean_distance(cluster_means, expand_dims(input, 1))

        if self.style == 'closest':
            _, pred = sample_costs.min(1)
            pred = cluster_classes[pred]
            acc = pred.eq(target).float()
            return torch.zeros(1), torch.zeros(1), pred, acc
        else:
            num_clusters = cluster_means.size()[0]

            # Sort the clusters by closest distance to sample
            sorted_sample_costs, indices = torch.sort(sample_costs)
            sorted_sample_costs = sorted_sample_costs.squeeze()
            indices = indices.type(GPU_LONG_DTYPE).squeeze()
            sorted_cluster_classes = cluster_classes[indices]

            # If L < num_clusters then lets only take the top L
            if self.L < num_clusters:
                sorted_sample_costs = sorted_sample_costs[:self.L]
                sorted_cluster_classes = sorted_cluster_classes[:self.L]
                num_clusters = self.L

            var_normalizer = -1 / (2 * self.variance ** 2)

            normalised_costs = torch.exp(var_normalizer * sorted_sample_costs).type(GPU_FLOAT_DTYPE)

            per_class_costs = torch.zeros(num_classes, num_clusters).type(GPU_FLOAT_DTYPE)
            per_class_costs = per_class_costs.scatter_(0, sorted_cluster_classes.unsqueeze(0), normalised_costs.unsqueeze(0))
            numerator = per_class_costs.sum(1)

            denominator = torch.sum(normalised_costs)

            epsilon = 1e-8

            probs = numerator / (denominator + epsilon)

            _, pred = probs.max(0)
            acc = pred.eq(target).float()

            return torch.zeros(1), torch.zeros(1), pred, acc


def expand_dims(var, dim=0):
    """ Is similar to [numpy.expand_dims](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html).
        var = torch.range(0, 9).view(-1, 2)
        torch.expand_dims(var, 0).size()
        # (1, 5, 2)
    """
    sizes = list(var.size())
    sizes.insert(dim, 1)
    return var.view(*sizes)


def comparison_mask(a_labels, b_labels):
    """Computes boolean mask for distance comparisons"""
    return torch.eq(expand_dims(a_labels, 1),
                    expand_dims(b_labels, 0))


def dynamic_partition(X, partitions, n_clusters):
    """Partitions the data into the number of cluster bins"""
    cluster_bin = torch.chunk(X, n_clusters)
    return cluster_bin


def compute_euclidean_distance(x, y):
    return torch.sum((x - y)**2, dim=2)
