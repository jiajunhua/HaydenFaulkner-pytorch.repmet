
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

from .rpn.rpn import RPN

from .roi_layers import ROIAlign, ROIPool

from .rcnn_target import RCNNTargetSampler


class FasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self,
                 output_size,
                 config):

        super(FasterRCNN, self).__init__()
        self.output_size = output_size

        self.class_agnostic = config.model.class_agnostic
        self.backbone_type = config.model.backbone.type
        self.pooling_mode = config.model.pooling_mode
        self.truncated = config.train.truncated

        # Get backbone network
        self.backbone_base_model, self.backbone_top_model, dout_base_model = self._init_backbone(type=config.model.backbone.type,
                                                                                     n_layers=config.model.backbone.n_layers,
                                                                                     pretrained=config.model.backbone.pretrained,
                                                                                     fixed_blocks=config.model.backbone.resnet_fixed_blocks)

        # Set up the classification layers
        if config.model.backbone.type == 'resnet':
            hid_dim = 2048
        elif config.model.backbone.type == 'vgg':
            hid_dim = 4096

        self.class_predictor = nn.Linear(hid_dim, self.output_size)
        if self.class_agnostic:
            self.box_predictor = nn.Linear(hid_dim, 4)
        else:
            self.box_predictor = nn.Linear(hid_dim, 4 * self.output_size)

        # define rpn
        self.rpn = RPN(config, dout_base_model,
                       channels=512,
                       stride=config.model.rpn.feat_stride,
                       anchor_scales=config.model.rpn.anchor_scales,
                       anchor_ratios=config.model.rpn.anchor_ratios)

        self.sampler = RCNNTargetSampler(nclasses=self.output_size,
                                         batch_size=config.train.batch_size,
                                         fg_fraction=config.train.fg_fraction,
                                         fg_thresh=config.train.fg_thresh,
                                         bg_thresh_high=config.train.bg_thresh_high,
                                         bg_thresh_low=config.train.bg_thresh_low,
                                         bbox_normalize_means=config.train.bbox_normalize_means,
                                         bbox_normalize_stds=config.train.bbox_normalize_stds,
                                         bbox_normalize_inside_weights=config.train.bbox_normalize_inside_weights)


        self.RCNN_roi_pool = ROIPool((config.model.pooling_size, config.model.pooling_size), 1.0/config.model.rpn.feat_stride)
        self.RCNN_roi_align = ROIAlign((config.model.pooling_size, config.model.pooling_size), 1.0/config.model.rpn.feat_stride, 0)

        self._init_weights()

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.backbone_base_model(im_data)

        # feed base feature map tp RPN to obtain rois
        if self.training:
            rpn_scores, rpn_bboxs, rpn_cls_scores, rpn_bbox_preds, anchors = self.rpn(base_feat, im_info, gt_boxes, num_boxes)

            gt_rois, rois, rois_label = self.sampler(rpn_bboxs, gt_boxes, num_boxes)

            rois = Variable(rois)
            rois_label = Variable(rois_label.view(-1).long())

        else:
            _, rpn_bboxs = self.rpn(base_feat, im_info, gt_boxes, num_boxes)


        # do roi pooling based on predicted rois
        if self.pooling_mode == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))
        elif self.pooling_mode == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        else:
            raise ValueError("Invalid pooling mode: {}".format(self.pooling_mode))

        # RCNN prediction
        # feed pooled features to top model
        top_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.box_predictor(top_feat)  # 1,84
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)  # 1,21,4
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4)) #1,1,6
            bbox_pred = bbox_pred_select.squeeze(1) #1,4

        # compute object classification probability
        cls_pred = self.class_predictor(top_feat)

        return gt_rois, rois, rois_label, cls_pred, bbox_pred, rpn_scores, rpn_bboxs, rpn_cls_scores, rpn_bbox_preds, anchors

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.rpn.rpn_Conv, 0, 0.01, self.truncated)
        normal_init(self.rpn.rpn_cls_score, 0, 0.01, self.truncated)
        normal_init(self.rpn.rpn_bbox_pred, 0, 0.01, self.truncated)
        normal_init(self.class_predictor, 0, 0.01, self.truncated)
        normal_init(self.box_predictor, 0, 0.001, self.truncated)

    def _init_backbone(self, type='resnet', n_layers=101, pretrained=False, fixed_blocks=None):

        # initialise the backbone and load weights
        if pretrained == 'caffe':
            if type == 'resnet' and n_layers == 101:
                backbone = models.resnet101(pretrained=False)
                model_path = 'data/pretrained_model/resnet101_caffe.pth'
            elif type == 'vgg' and n_layers == 16:
                backbone = models.vgg16(pretrained=False)
                model_path = 'data/pretrained_model/vgg16_caffe.pth'
            else:
                ValueError

            print("Loading pretrained weights from %s" % model_path)
            state_dict = torch.load(model_path)
            backbone.load_state_dict({k: v for k, v in state_dict.items() if k in backbone.state_dict()})

        elif type == 'resnet' and n_layers == 101:
            backbone = models.resnet101(pretrained=pretrained)

        if type == 'resnet':
            # Make the backbone components
            backbone_base_model = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                          backbone.maxpool, backbone.layer1, backbone.layer2, backbone.layer3)

            backbone_top_model = nn.Sequential(backbone.layer4)

            dout_base_model = 1024

            # Fix the parameters of the backbone
            for p in backbone_base_model[0].parameters(): p.requires_grad = False
            for p in backbone_base_model[1].parameters(): p.requires_grad = False

            assert (0 <= fixed_blocks < 4)
            if fixed_blocks >= 3:
                for p in backbone_base_model[6].parameters(): p.requires_grad = False
            if fixed_blocks >= 2:
                for p in backbone_base_model[5].parameters(): p.requires_grad = False
            if fixed_blocks >= 1:
                for p in backbone_base_model[4].parameters(): p.requires_grad = False

            def set_bn_fix(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    for p in m.parameters(): p.requires_grad = False

            backbone_base_model.apply(set_bn_fix)
            backbone_top_model.apply(set_bn_fix)

        elif type == 'vgg':
            # Make the backbone components
            backbone_base_model = nn.Sequential(*list(backbone.features._modules.values())[:-1])
            backbone_top_model = nn.Sequential(*list(backbone.classifier._modules.values())[:-1])

            dout_base_model = 512

            # Fix the parameters of the backbone
            # Fix the layers before conv3:
            for layer in range(10):
                for p in backbone_base_model[layer].parameters(): p.requires_grad = False

        return backbone_base_model, backbone_top_model, dout_base_model

    def train(self, mode=True): # for resnet only
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode and self.backbone_type == 'resnet':
            # Set fixed blocks to be in eval mode
            self.backbone_base_model.eval()
            self.backbone_base_model[5].train()
            self.backbone_base_model[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.backbone_base_model.apply(set_bn_eval)
            self.backbone_top_model.apply(set_bn_eval)

    def _head_to_tail(self, pool5):

        if self.backbone_type == 'resnet':
            fc7 = self.backbone_top_model(pool5).mean(3).mean(2)

        elif self.backbone_type == 'vgg':
            pool5_flat = pool5.view(pool5.size(0), -1)
            fc7 = self.backbone_top_model(pool5_flat)

        return fc7

if __name__ == "__main__":
    # use this for debugging and checks
    from utils.debug import set_working_dir
    from config.config import config
    import matplotlib.pyplot as plt

    # set the working directory as appropriate
    set_working_dir()

    # load the dataset
    model = FasterRCNN(list(range(100)), config=config)