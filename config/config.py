"""
Configuration defaults specification file
"""
import yaml
import numpy as np
from easydict import EasyDict as edict

# Project Setup
config = edict()
config.project = 'pytorch.repmet'
config.seed = 7
config.gpus = '0'

# Shared Defaults
config.run_type = None
config.run_id = None


config.emb_dim = 256

# Model Defaults
config.model = edict()
config.model.root_dir = 'models'
config.model.type = None
config.model.id = None

config.model.emb_size = '' # num classes when not emb
config.model.dist = 'euc'

config.model.backbone = edict()
config.model.backbone.type = 'resnet'
config.model.backbone.n_layers = 101
config.model.backbone.pretrained = True
config.model.backbone.resnet_fixed_blocks = 1

config.model.rpn = edict()
config.model.rpn.anchor_scales = [8, 16, 32]
config.model.rpn.anchor_ratios = [0.5, 1, 2]
config.model.rpn.feat_stride = 16.0

# Detection
config.model.max_n_gt_boxes = 20
config.model.class_agnostic = False
config.model.pooling_mode = 'align'  # crop default in orig code but not imported so suggested never used
config.model.pooling_size = 7

# Dataset Defaults
config.dataset = edict()
config.dataset.root_dir = 'data'
config.dataset.name = None
config.dataset.id = None
config.dataset.classes = ''

# detection
config.dataset.use_flipped = True
config.dataset.use_difficult = False

# Train Defaults
config.train = edict()
config.train.sampler = None
config.train.loss = None

config.train.checkpoint_every = 0  # 0 is never

config.train.for_bs = 64  # the batch size for forward pass for building clusters (magnet) or reps (repmet), lower if running out of mem

config.train.epochs = None

config.train.optimizer = 'sgd'
config.train.learning_rate = 0.001
config.train.lr_scheduler_gamma = ''
config.train.lr_scheduler_step = ''
config.train.momentum = 0.9

config.train.weight_decay = 0.0005  # Weight decay, for regularization
config.train.bias_decay = False  # Whether to have weight decay on bias as well

config.train.double_bias = True  # Whether to double the learning rate for bias

config.train.episodes = ''
# protos
config.train.categories_per_epi = ''
config.train.support_per_epi = ''
config.train.query_per_epi = ''
# magnet + repmet
config.train.k = ''
config.train.m = ''
config.train.d = ''
# repmet
config.train.alpha = ''
config.train.sigma = ''

# detection
config.train.scales = (600,)  # Scale to use during testing (can list multiple scales) The scale is the pixel size of an image's shortest side
config.train.max_size = 1000  # Max pixel size of the longest side of a scaled input image


config.train.img_per_batch = 1  # Images to use per minibatch
config.train.batch_size = 128  # Minibatch size (number of regions of interest [ROIs])

config.train.use_all_gt = True  # For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''

config.train.truncated = False  # Whether to initialize the weights with truncated normal distribution

config.train.fg_fraction = 0.25  # Fraction of minibatch that is labeled foreground (i.e. class > 0)
config.train.fg_thresh = 0.5  # Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
config.train.bg_thresh_high = 0.5  # Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))
config.train.bg_thresh_low = 0.1

# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
config.train.bbox_normalize_targets_precomputed = True
config.train.bbox_normalize_means = (0.0, 0.0, 0.0, 0.0)
config.train.bbox_normalize_stds = (0.1, 0.1, 0.2, 0.2)
config.train.bbox_normalize_inside_weights = (1.0, 1.0, 1.0, 1.0)

config.train.rpn = edict()
config.train.rpn.pre_nms_top_n = 12000
config.train.rpn.post_nms_top_n = 2000
config.train.rpn.nms_thresh = 0.7
config.train.rpn.min_size = 8
config.train.rpn.batch_size = 256  # Total number of examples
config.train.rpn.clobber_positives = False  # If an anchor statisfied by positive and negative conditions set to negative
config.train.rpn.fg_fraction = 0.5  # Max number of foreground examples fg_rois_per_image=this*rpn.batch_size ie 256*0.5
config.train.rpn.positive_overlap = 0.7  # IOU >= thresh: positive example
config.train.rpn.negative_overlap = 0.3  # IOU < thresh: negative example

# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
config.train.rpn.positive_weight = -1.0

# Deprecated (outside weights)
config.train.rpn.bbox_inside_weights = (1.0, 1.0, 1.0, 1.0)


# Validation Defaults
config.val = edict()
config.val.every = 0  # 0 is never

config.val.sampler = None
config.val.loss = None

config.val.episodes = ''

# protos
config.val.categories_per_epi = ''
config.val.support_per_epi = ''
config.val.query_per_epi = ''
# magnet
config.val.L = ''
config.val.style = 'magnet'  # or 'closest'
# repmet
config.val.m = ''
config.val.d = ''
config.val.alpha = ''
config.val.sigma = ''


# Visualisation Defaults
config.vis = edict()
config.vis.every = 0  # 0 is never
config.vis.plot_embed_every = 0  # 0 is never
config.vis.test_plot_embed_every = 0  # 0 is never

# config.train.dml = False  # Use embedding networks? Baselines are false  # TODO does this go in model config?

# Test Defaults
config.test = edict()

config.test.split = 'test'

config.test.resume_from = 'B'  # B is best, L is latest, or define own path

config.test.sampler = None
config.test.loss = None

config.test.episodes = ''

# protos
config.test.categories_per_epi = ''
config.test.support_per_epi = ''
config.test.query_per_epi = ''
# repmet
config.test.m = ''
config.test.d = ''
config.test.alpha = ''
config.test.sigma = ''

config.test.rpn = edict()
config.test.rpn.pre_nms_top_n = 6000
config.test.rpn.post_nms_top_n = 300
config.test.rpn.nms_thresh = 0.7
config.test.rpn.min_size = 16

# def update_config(config_file):
#     with open(config_file) as f:
#         exp_config = edict(yaml.load(f))
#         for k, v in exp_config.items():
#             if k in config:
#                 if isinstance(v, dict):
#                     if k == 'TRAIN':
#                         if 'BBOX_WEIGHTS' in v:
#                             v['BBOX_WEIGHTS'] = np.array(v['BBOX_WEIGHTS'])
#                     elif k == 'network':
#                         if 'PIXEL_MEANS' in v:
#                             v['PIXEL_MEANS'] = np.array(v['PIXEL_MEANS'])
#                     for vk, vv in v.items():
#                         config[k][vk] = vv
#                 else:
#                     if k == 'SCALES':
#                         config[k][0] = (tuple(v))
#                     else:
#                         config[k] = v
#             else:
#                 raise ValueError("key (%s) must exist in config.py" % k)

def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        recursive_update(exp_config, c=config)

        # for k, v in exp_config.items():
        #     if k in config:
        #         if isinstance(v, dict):
        #             if k == 'TRAIN':
        #                 if 'BBOX_WEIGHTS' in v:
        #                     v['BBOX_WEIGHTS'] = np.array(v['BBOX_WEIGHTS'])
        #             elif k == 'network':
        #                 if 'PIXEL_MEANS' in v:
        #                     v['PIXEL_MEANS'] = np.array(v['PIXEL_MEANS'])
        #             for vk, vv in v.items():
        #                 config[k][vk] = vv
        #         else:
        #             if k == 'SCALES':
        #                 config[k][0] = (tuple(v))
        #             else:
        #                 config[k] = v
        #     else:
        #         raise ValueError("key (%s) must exist in config.py" % k)

def recursive_update(in_config, c):
    for ki, vi in in_config.items():
        if isinstance(vi, edict):
            recursive_update(vi, c[ki])
        else:
            c[ki] = vi

def check_config(in_config, k=''):
    # recursive function to check for no Nones...
    # All default Nones need to be specified in experimental .yaml files
    for ki, vi in in_config.items():
        if isinstance(vi, edict):
            check_config(vi, k+'.'+ki)
        elif vi is None:
            raise ValueError("%s must be specified in the .yaml config file" % (k+'.'+ki))
        elif vi == '':
            in_config[ki] = None  # todo consider removing unset but some need to be none to spec default...
