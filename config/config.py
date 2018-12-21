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
config.gpus = ''

# Shared Defaults
config.run_type = None
config.run_id = None


config.emb_dim = 256

# Model Defaults
config.model = edict()
config.model.root_dir = 'models'
config.model.type = None
config.model.id = None

config.model.backbone = edict()
config.model.backbone.name = ''  # What model spec to use for the backbone, if backbone nec
config.model.backbone.out_layer = ''  # What layer do we take from the backbone net

# Dataset Defaults
config.dataset = edict()
config.dataset.root_dir = 'data'
config.dataset.name = None
config.dataset.id = None

# Train Defaults
config.train = edict()
config.train.sampler = None
config.train.loss = None

config.train.checkpoint_every = 0  # 0 is never

config.train.epochs = None
config.train.learning_rate = None
config.train.lr_scheduler_gamma = ''
config.train.lr_scheduler_step = ''

config.train.episodes = ''
# protos
config.train.categories_per_epi = ''
config.train.support_per_epi = ''
config.train.query_per_epi = ''
# magnet
config.train.k = ''
config.train.m = ''
config.train.d = ''

# Validation Defaults
config.val = edict()
config.val.every = 0  # 0 is never

config.val.sampler = None
config.val.loss = None

# protos
config.val.episodes = ''
config.val.categories_per_epi = ''
config.val.support_per_epi = ''
config.val.query_per_epi = ''
# magnet
config.val.L = ''
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
config.test.categories_per_epi = ''
config.test.support_per_epi = ''
config.test.query_per_epi = ''


# Classification Specific Defaults
# parser.add_argument('--set_name', required=True, help='dataset name', default='mnist')
# parser.add_argument('--model_name', required=True, help='model name', default='mnist_default')
# parser.add_argument('--loss_type', required=True, help='magnet, repmet, repmet2', default='repmet2')
# parser.add_argument('--m', required=True, help='number of clusters per batch', default=8, type=int)
# parser.add_argument('--d', required=True, help='number of samples per cluster per batch', default=8, type=int)
# parser.add_argument('--k', required=True, help='number of clusters per class', default=3, type=int)
# parser.add_argument('--alpha', required=True, help='cluster margin', default=1.0, type=int)
# parser.add_argument('--n_iterations', required=False, help='number of iterations to perform', default=1000, type=int)
# parser.add_argument('--net_learning_rate', required=False, help='the learning rate for the net', default=0.0001,
#                     type=float)
# parser.add_argument('--cluster_learning_rate', required=False,
#                     help='the learning rate for the modes (centroids), if -1 will use single optimiser for both net and modes',
#                     default=0.001, type=float)
# parser.add_argument('--chunk_size', required=False,
#                     help='the chunk/batch size for calculating embeddings (lower for less mem)', default=32, type=int)
# parser.add_argument('--refresh_clusters', required=False,
#                     help='refresh the clusters every ? iterations or on these iterations (int or list or ints)',
#                     default=50)
# parser.add_argument('--calc_acc_every', required=False, help='calculate the accuracy every ? iterations', default=10,
#                     type=int)
# parser.add_argument('--load_latest', required=False, help='load a model if presaved', default=True)
# parser.add_argument('--save_every', required=False, help='save the model every ? iterations', default=200, type=int)
# parser.add_argument('--save_path', required=False, help='where to save the model_definitions',
#                     default=configs.general.paths.models)
# parser.add_argument('--plot_every', required=False, help='plot graphs every ? iterations', default=100, type=int)
# parser.add_argument('--plots_path', required=False, help='where to save the plots',
#                     default=configs.general.paths.graphing)
# parser.add_argument('--plots_ext', required=False, help='.png/.pdf', default='.png')
# parser.add_argument('--n_plot_samples', required=False, help='plot ? samples per class', default=10, type=int)
# parser.add_argument('--n_plot_classes', required=False, help='plot ? classes', default=10, type=int)

# Detection Specific Defaults






# config.MXNET_VERSION = ''
# config.output_path = ''
# config.symbol = ''
# config.gpus = ''
# config.FRAMES = 1
# config.CLASS_AGNOSTIC = True
# config.MODEL_BACKGROUND = False
# config.DML = True
# config.DML_SUBNET_fc2 = True
# config.DML_SUBNET_p5 = False
# config.EMBEDDING_SIZE = 256
# config.SCALES = [(600, 1000)]  # first is scale (the shorter side); second is max size
# config.TEST_SCALES = [(600, 1000)]
# # default training
# config.default = edict()
# config.default.frequent = 20
# config.default.kvstore = 'device'
#
# # network related params
# config.network = edict()
# config.network.pretrained = ''
# config.network.pretrained_epoch = 0
# config.network.PIXEL_MEANS = np.array([0, 0, 0])
# config.network.IMAGE_STRIDE = 0
# config.network.RPN_FEAT_STRIDE = 16
# config.network.RCNN_FEAT_STRIDE = 16
# config.network.FIXED_PARAMS = ['gamma', 'beta']
# config.network.FIXED_PARAMS_SHARED = ['gamma', 'beta']
# config.network.ANCHOR_SCALES = (8, 16, 32)
# config.network.ANCHOR_RATIOS = (0.5, 1, 2)
# config.network.NUM_ANCHORS = len(config.network.ANCHOR_SCALES) * len(config.network.ANCHOR_RATIOS)
#
# # dataset related params
# config.dataset = edict()
# config.dataset.dataset = 'PascalVOC'
# config.dataset.image_set = '2007_trainval'
# config.dataset.test_image_set = '2007_test'
# config.dataset.root_path = './data_loading'
# config.dataset.dataset_path = './data_loading/VOCdevkit'
# config.dataset.classes_path = ""
# config.dataset.test_classes_path = ""
# config.dataset.NUM_CLASSES = 21
#
#
# config.TRAIN = edict()
#
# config.TRAIN.lr = 0
# config.TRAIN.kmeans = 0
# config.TRAIN.k = 5
# config.TRAIN.lr_step = ''
# config.TRAIN.lr_factor = 0.1
# config.TRAIN.warmup = False
# config.TRAIN.warmup_lr = 0
# config.TRAIN.warmup_step = 0
# config.TRAIN.momentum = 0.9
# config.TRAIN.wd = 0.0005
# config.TRAIN.begin_epoch = -1
# config.TRAIN.begin_iter = -1
# config.TRAIN.load_data = True
# config.TRAIN.batch_load = False
# config.TRAIN.batch_save = 200000
# config.TRAIN.end_epoch = 0
# config.TRAIN.model_prefix = ''
#
# config.TRAIN.ALTERNATE = edict()
# config.TRAIN.ALTERNATE.RPN_BATCH_IMAGES = 0
# config.TRAIN.ALTERNATE.RCNN_BATCH_IMAGES = 0
# config.TRAIN.ALTERNATE.rpn1_lr = 0
# config.TRAIN.ALTERNATE.rpn1_lr_step = ''    # recommend '2'
# config.TRAIN.ALTERNATE.rpn1_epoch = 0       # recommend 3
# config.TRAIN.ALTERNATE.rfcn1_lr = 0
# config.TRAIN.ALTERNATE.rfcn1_lr_step = ''   # recommend '5'
# config.TRAIN.ALTERNATE.rfcn1_epoch = 0      # recommend 8
# config.TRAIN.ALTERNATE.rpn2_lr = 0
# config.TRAIN.ALTERNATE.rpn2_lr_step = ''    # recommend '2'
# config.TRAIN.ALTERNATE.rpn2_epoch = 0       # recommend 3
# config.TRAIN.ALTERNATE.rfcn2_lr = 0
# config.TRAIN.ALTERNATE.rfcn2_lr_step = ''   # recommend '5'
# config.TRAIN.ALTERNATE.rfcn2_epoch = 0      # recommend 8
# # optional
# config.TRAIN.ALTERNATE.rpn3_lr = 0
# config.TRAIN.ALTERNATE.rpn3_lr_step = ''    # recommend '2'
# config.TRAIN.ALTERNATE.rpn3_epoch = 0       # recommend 3
#
# # whether resume training
# config.TRAIN.RESUME = False
# # whether flip image
# config.TRAIN.FLIP = True
# # whether shuffle image
# config.TRAIN.SHUFFLE = True
# # use emb loss or just CE
# config.TRAIN.USE_EMB_LOSS = True
#
# # what to ignore if anything
# config.TRAIN.CE_ignore = True
# config.TRAIN.CE_ignore = True
# config.TRAIN.CE_ignore_other = False
# config.TRAIN.CE_norm = True
#
# # whether use OHEM
# config.TRAIN.ENABLE_OHEM = False
# # size of images for each device, 2 for rcnn, 1 for rpn and e2e
# config.TRAIN.BATCH_IMAGES = 2
# # e2e changes behavior of anchor loader and metric
# config.TRAIN.END2END = False
# # group images with similar aspect ratio
# config.TRAIN.ASPECT_GROUPING = True
#
# # R-CNN
# # rcnn rois batch size
# config.TRAIN.BATCH_ROIS = 128
# config.TRAIN.BATCH_ROIS_OHEM = 128
# # rcnn rois sampling params
# config.TRAIN.FG_FRACTION = 0.25
# config.TRAIN.FG_THRESH = 0.5
# config.TRAIN.BG_THRESH_HI = 0.5
# config.TRAIN.BG_THRESH_LO = 0.0
# # rcnn bounding box regression params
# config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
# config.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])
#
# # RPN anchor loader
# # rpn anchors batch size
# config.TRAIN.RPN_BATCH_SIZE = 256
# # rpn anchors sampling params
# config.TRAIN.RPN_FG_FRACTION = 0.5
# config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# config.TRAIN.RPN_CLOBBER_POSITIVES = False
# # rpn bounding box regression params
# config.TRAIN.RPN_BBOX_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
#
# # used for end2end training
# # RPN proposal
# config.TRAIN.CXX_PROPOSAL = True
# config.TRAIN.RPN_NMS_THRESH = 0.7
# config.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# config.TRAIN.RPN_POST_NMS_TOP_N = 2000
# config.TRAIN.RPN_MIN_SIZE = config.network.RPN_FEAT_STRIDE
# # approximate bounding box regression
# config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = False
# config.TRAIN.BBOX_MEANS = (0.0, 0.0, 0.0, 0.0)
# config.TRAIN.BBOX_STDS = (0.1, 0.1, 0.2, 0.2)
#
# config.TRAIN.STRICTNESS = 'strict'
# config.TRAIN.REP_MMM = 'min'
#
# config.TRAIN.alpha = 1.0
#
# config.TEST = edict()
#
# # R-CNN testing
# # use rpn to generate proposal
# config.TEST.HAS_RPN = False
# # size of images for each device
# config.TEST.BATCH_IMAGES = 1
#
# # RPN proposal
# config.TEST.CXX_PROPOSAL = True
# config.TEST.RPN_NMS_THRESH = 0.7
# config.TEST.RPN_PRE_NMS_TOP_N = 6000
# config.TEST.RPN_POST_NMS_TOP_N = 300
# config.TEST.RPN_MIN_SIZE = config.network.RPN_FEAT_STRIDE
#
# # RPN generate proposal
# config.TEST.PROPOSAL_NMS_THRESH = 0.7
# config.TEST.PROPOSAL_PRE_NMS_TOP_N = 20000
# config.TEST.PROPOSAL_POST_NMS_TOP_N = 2000
# config.TEST.PROPOSAL_MIN_SIZE = config.network.RPN_FEAT_STRIDE
#
# # RCNN nms
# config.TEST.NMS = 0.3
#
# config.TEST.max_per_image = 300
#
# # Test Model Epoch
# config.TEST.test_epoch = 0
# config.TEST.test_batch = 0
#
# config.TEST.USE_SOFTNMS = False
#
#
# config.TEST.n_shot = 5
# config.TEST.m_way = 5
# config.TEST.num_episodes = 2000


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    if k == 'TRAIN':
                        if 'BBOX_WEIGHTS' in v:
                            v['BBOX_WEIGHTS'] = np.array(v['BBOX_WEIGHTS'])
                    elif k == 'network':
                        if 'PIXEL_MEANS' in v:
                            v['PIXEL_MEANS'] = np.array(v['PIXEL_MEANS'])
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("key (%s) must exist in config.py" % k)


def check_config(in_config, k=''):
    # recursive function to check for no Nones...
    # All default Nones need to be specified in experimental .yaml files
    for ki, vi in in_config.items():
        if isinstance(vi, edict):
            check_config(vi, k+'.'+ki)
        elif vi is None:
            raise ValueError("%s must be specified in the .yaml config file" % (k+'.'+ki))
