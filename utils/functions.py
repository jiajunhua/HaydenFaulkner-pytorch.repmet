import torch


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


def dynamic_partition(X, n_clusters):
    """Partitions the data into the number of cluster bins"""
    cluster_bin = torch.chunk(X, n_clusters)
    return cluster_bin


def compute_euclidean_distance(x, y):
    return torch.sum((x - y)**2, dim=2)


def make_one_hot(labels, n_classes):
    """

    :param labels: the labels in int form
    :param n_classes: the number of classes
    :return: a one hot vector with these class labels
    """
    one_hot = torch.zeros((labels.shape[-1], n_classes))
    return one_hot.scatter_(1, torch.unsqueeze(labels, 1).long(), 1).byte()
