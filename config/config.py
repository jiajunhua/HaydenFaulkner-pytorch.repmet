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

config.model.emb_size = ''
config.model.dist = 'euc'

config.model.backbone = edict()
config.model.backbone.name = ''  # What model spec to use for the backbone, if backbone nec
config.model.backbone.out_layer = ''  # What layer do we take from the backbone net

# Detection
config.model.max_n_gt_boxes = 20

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
config.train.learning_rate = None
config.train.lr_scheduler_gamma = ''
config.train.lr_scheduler_step = ''

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
        elif vi == '':
            in_config[ki] = None  # todo consider removing unset but some need to be none to spec default...
