import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from .proposal_layer import RPNProposal
from .generate_anchors import generate_anchor_bases


class RPN(nn.Module):
    """ region proposal network """

    def __init__(self, config, din, channels, stride, anchor_scales, anchor_ratios):

        super(RPN, self).__init__()

        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.stride = stride

        # define the convrelu layers processing input feature map
        self.rpn_Conv = nn.Conv2d(self.din, channels, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2  # 2(bg/fg) * 9 (anchors)
        self.rpn_cls_score = nn.Conv2d(channels, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4  # 4(coords) * 9 (anchors)
        self.rpn_bbox_pred = nn.Conv2d(channels, self.nc_bbox_out, 1, 1, 0)

        # generate the anchor bases
        self.anchor_bases = generate_anchor_bases(base_size=16, scales=np.array(anchor_scales), ratios=np.array(anchor_ratios))  # todo consider putting outside rpn?

        # define proposal layer
        self.region_proposaler = RPNProposal(stride=self.stride,
                                             anchor_bases=self.anchor_bases,
                                             pre_nms_top_n={'TRAIN': config.train.rpn.pre_nms_top_n,
                                                            'TEST': config.test.rpn.pre_nms_top_n},
                                             post_nms_top_n={'TRAIN': config.train.rpn.post_nms_top_n,
                                                             'TEST': config.test.rpn.post_nms_top_n},
                                             nms_thresh={'TRAIN': config.train.rpn.nms_thresh,
                                                         'TEST': config.test.rpn.nms_thresh},
                                             min_size={'TRAIN': config.train.rpn.min_size,
                                                       'TEST': config.test.rpn.min_size})

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.rpn_Conv(base_feat), inplace=True)

        # get rpn classification score
        rpn_cls_score = self.rpn_cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.rpn_bbox_pred(rpn_conv1)

        # proposal layer
        rpn_scores, rpn_bbox, anchors = self.region_proposaler(rpn_cls_prob.data, rpn_bbox_pred.data, im_info)

        if self.training:
            return rpn_scores, rpn_bbox, rpn_cls_score, rpn_bbox_pred, anchors

        return rpn_scores, rpn_bbox