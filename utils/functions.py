import torch
import torch.nn.functional as F


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


def make_one_hot(labels, n_classes):
    """

    :param labels: the labels in int form
    :param n_classes: the number of classes
    :return: a one hot vector with these class labels
    """
    one_hot = torch.zeros((labels.shape[-1], n_classes))
    return one_hot.scatter_(1, torch.unsqueeze(labels, 1).long().cpu(), 1).byte()


def euclidean_distance(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception("size mismatch")

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def cosine_distance(x, y):
    # x: N x D
    # y: M x D
    d = x.size(1)
    if d != y.size(1):
        raise Exception("size mismatch")

    x = x.transpose(0, 1).unsqueeze(0)
    y = y.unsqueeze(2)
    return - (F.cosine_similarity(y, x)-1).transpose(0, 1)  # (range of 0 - 2)


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):

    # used in rpn.rpn and faster_rcnn
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box