import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from utils.functions import _smooth_l1_loss
from model_definitions.detectors.faster_rcnn.rpn.rpn_target import RPNTargetGenerator
from model_definitions.detectors.faster_rcnn.rcnn_target import RCNNTargetGenerator


class DetectionLoss(nn.Module):

    def __init__(self, config):
        super(DetectionLoss, self).__init__()

        self.get_rpn_anchor_targets = RPNTargetGenerator(rpn_batch_size=config.train.rpn.batch_size,
                                                         positive_overlap=config.train.rpn.positive_overlap,
                                                         negative_overlap=config.train.rpn.negative_overlap,
                                                         fg_fraction=config.train.rpn.fg_fraction,
                                                         clobber_positives=config.train.rpn.clobber_positives,
                                                         n_base_anchors=len(config.model.rpn.anchor_scales)*len(config.model.rpn.anchor_ratios), # 9
                                                         positive_weight=config.train.rpn.positive_weight,
                                                         bbox_inside_weights=config.train.rpn.bbox_inside_weights)

        self.get_rcnn_proposal_targets = RCNNTargetGenerator(bbox_normalize_targets_precomputed=config.train.bbox_normalize_targets_precomputed,
                                                             bbox_normalize_means=config.train.bbox_normalize_means,
                                                             bbox_normalize_stds=config.train.bbox_normalize_stds,
                                                             bbox_normalize_inside_weights=config.train.bbox_normalize_inside_weights)

    def forward(self, input, target):
        gt_rois = input[0]
        rois = input[1]
        rois_label = input[2]

        cls_pred = input[3]
        bbox_pred = input[4]

        rpn_scores = input[5]
        rpn_bboxs = input[6]

        rpn_cls_scores = input[7]
        rpn_bbox_preds = input[8]

        anchors = input[9]

        gt_boxes = target[0]
        num_boxes = target[1]
        im_info = target[2]

        assert gt_boxes is not None

        batch_size = gt_boxes.size(0)

        # Compute RPN targets
        rpn_cls_targets, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            self.get_rpn_anchor_targets(rpn_cls_scores.data, gt_boxes, im_info, num_boxes, anchors)

        # compute bbox classification loss
        rpn_cls_scores_reshape = rpn_cls_scores.view(batch_size, 2, int(np.round(rpn_cls_scores.size(1)*rpn_cls_scores.size(2)/2)), -1)  # (1,18,75,38) > (1,2,513,38)
        rpn_cls_scores = rpn_cls_scores_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)  # (1, 19494, 2)
        rpn_cls_targets = rpn_cls_targets.view(batch_size, -1)

        rpn_keep = Variable(rpn_cls_targets.view(-1).ne(-1).nonzero().view(-1))
        rpn_cls_scores = torch.index_select(rpn_cls_scores.view(-1, 2), 0, rpn_keep)
        rpn_cls_targets = torch.index_select(rpn_cls_targets.view(-1), 0, rpn_keep.data)
        rpn_cls_targets = Variable(rpn_cls_targets.long())
        rpn_loss_cls = F.cross_entropy(rpn_cls_scores, rpn_cls_targets)
        fg_cnt = torch.sum(rpn_cls_targets.data.ne(0))

        # compute bbox regression loss
        rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
        rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
        rpn_bbox_targets = Variable(rpn_bbox_targets)

        rpn_loss_box = _smooth_l1_loss(rpn_bbox_preds, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3])


        # Compute RCNN targets
        rois_target, rois_inside_ws, rois_outside_ws = self.get_rcnn_proposal_targets(gt_rois, rois, rois_label)

        # rois_label = Variable(rois_label.view(-1).long())
        rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
        rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
        rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))


        # compute bbox classification loss
        cls_prob = F.softmax(cls_pred, 1)
        rcnn_loss_cls = F.cross_entropy(cls_pred, rois_label)

        # compute bbox regression loss
        rcnn_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        # Add the losses together
        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + rcnn_loss_cls.mean() + rcnn_loss_bbox.mean()

        # Calc RPN Acc
        rpn_label = rpn_cls_targets
        rpn_weight = rpn_cls_targets>=0  # for the ignore -1 flag
        rpn_cls_logits = rpn_cls_scores

        num_inst = rpn_weight.sum()

        pred_label = torch.argmax(rpn_cls_logits, dim=1, keepdim=True).squeeze()

        num_acc = ((pred_label == rpn_label) * rpn_weight).sum()
        rpn_acc = num_acc / num_inst


        # Calc RCNN Acc
        pred_label = torch.argmax(cls_pred, dim=-1)
        num_acc = torch.sum(pred_label == rois_label)

        rcnn_acc = num_acc / rois_label.shape[0]

        return loss, rpn_loss_cls.mean(), rpn_loss_box.mean(), rcnn_loss_cls.mean(), rcnn_loss_bbox.mean(), rpn_acc, rcnn_acc
