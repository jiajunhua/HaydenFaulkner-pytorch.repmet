import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class DetectionLoss(nn.Module):

    def __init__(self):
        super(DetectionLoss, self).__init__()

    def forward(self, input, target):
        rois = input[0]
        cls_prob = input[1]
        bbox_pred = input[2]
        rpn_loss_cls = input[3]
        rpn_loss_box = input[4]
        RCNN_loss_cls = input[5]
        RCNN_loss_bbox = input[6]
        rois_label = input[7]


        sample_losses = rpn_loss_cls + rpn_loss_box + RCNN_loss_cls + RCNN_loss_bbox


        soft_probs = numerator / (denominator + epsilon) + epsilon

        _, pred = soft_probs.max(1)
        acc = pred.eq(target.squeeze()).float().mean()

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # if cfg.TEST.BBOX_REG:
        #     # Apply bounding-box regression deltas
        #     box_deltas = bbox_pred.data
        #     if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        #         # Optionally normalize targets by a precomputed mean and stdev
        #         if args.class_agnostic:
        #             box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
        #                          + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        #             box_deltas = box_deltas.view(1, -1, 4)
        #         else:
        #             box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
        #                          + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        #             box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))
        #
        #     pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        #     pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        # else:
        #     # Simply repeat the boxes, once for each class
        #     pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        #
        # pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()


        return total_loss, losses, pred, acc
    
    
    def agnostic_box_iou(self, gts, dts, ovthresh=0.5):

        nd = len(dts)  # number of detections
        ng = len(gts)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        if dts.shape[0] > 0:
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            dts = dts[sorted_ind, :]

            # go down dets and mark TPs and FPs
            for d in range(nd):
                dt = dts[d, :].astype(float)
                ovmax = -np.inf

                if gts.size > 0:
                    # compute overlaps
                    overlaps = self.iou(gts, dt)
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / ng
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.voc_ap(rec, prec, False)
        return rec, prec, ap


    def iou(self, gts, dt):
        ixmin = np.maximum(gts[:, 0], dt[0])
        iymin = np.maximum(gts[:, 1], dt[1])
        ixmax = np.minimum(gts[:, 2], dt[2])
        iymax = np.minimum(gts[:, 3], dt[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((dt[2] - dt[0] + 1.) * (dt[3] - dt[1] + 1.) +
               (gts[:, 2] - gts[:, 0] + 1.) *
               (gts[:, 3] - gts[:, 1] + 1.) - inters)

        return inters / uni

    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
