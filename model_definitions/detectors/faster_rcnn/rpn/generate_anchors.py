# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
#
# Modified by Hayden Faulkner
# --------------------------------------------------------

import numpy as np


def generate_anchors(stride, base_size, ratios, scales, feat_size):
    """
    Pre-generate all anchors

    :param stride:
    :param base_size:
    :param ratios:
    :param scales:
    :param feat_size:
    :return:
    """

    anchor_bases = generate_anchor_bases(base_size, ratios, scales)  # gluon style
    anchors = shift_anchor_bases(anchor_bases, stride, feat_size)

    return anchors


def generate_anchor_bases(base_size, ratios, scales):
    """
    Generate all anchors bases

    :param base_size:
    :param ratios:
    :param scales:
    :return:
    """
    # generate same shapes on every location
    px, py = (base_size - 1) * 0.5, (base_size - 1) * 0.5
    anchor_bases = []
    for r in ratios:
        for s in scales:
            size = base_size * base_size / r
            ws = np.round(np.sqrt(size))
            w = (ws * s - 1) * 0.5
            h = (np.round(ws * r) * s - 1) * 0.5
            anchor_bases.append([px - w, py - h, px + w, py + h])
    anchor_bases = np.array(anchor_bases)  # (N, 4)

    return anchor_bases


def shift_anchor_bases(anchor_bases,stride, feat_size):
    """
    Shift anchor bases with the strides across the feat size

    :param anchor_bases:
    :param stride:
    :param feat_size:
    :return:
    """
    # propagete to all locations by shifting offsets
    height, width = feat_size
    offset_x = np.arange(0, width * stride, stride)
    offset_y = np.arange(0, height * stride, stride)
    offset_x, offset_y = np.meshgrid(offset_x, offset_y)
    offsets = np.stack((offset_x.ravel(), offset_y.ravel(),
                        offset_x.ravel(), offset_y.ravel()), axis=1)
    # broadcast_add (1, N, 4) + (M, 1, 4)
    anchors = (anchor_bases.reshape((1, -1, 4)) + offsets.reshape((-1, 1, 4)))
    anchors = anchors.reshape((1, anchors.shape[0]*anchors.shape[1], -1)).astype(np.float32)

    return anchors

