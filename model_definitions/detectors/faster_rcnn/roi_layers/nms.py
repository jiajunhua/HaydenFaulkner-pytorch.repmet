# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch  # Need this here so references correct torch imp
from roi_layers import _C

nms = _C.nms
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""