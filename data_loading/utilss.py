# """
# inspired by https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0
#
# particularly lib/roi_data_layer/roidb.py
# """
# import numpy as np
#
# def filter_dataset(dataset):
#     # todo if dataset written correctly it should never pick up any samples with 0 boxes
#     """
#     remove images (data samples) that don't have any gt boxes
#     :param dataset: non filtered data
#     :return: filtered data
#     """
#     print('before filtering, there are %d images...' % (len(dataset.data)))
#     for sample_id in dataset.data.keys():
#         if len(dataset.data[sample_id]['boxes']) == 0:
#             del dataset.data[sample_id]
#
#     print('after filtering, there are %d images...' % (len(dataset.data)))
#
#     return dataset
