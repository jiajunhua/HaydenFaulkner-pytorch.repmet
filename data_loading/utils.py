"""
inspired by https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0

particularly lib/roi_data_layer/roidb.py
"""
import numpy as np


def prepare_dataset(dataset):
    """Enrich the dataset's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """

    for sample_id in dataset.data.keys():
        dataset.data[sample_id]['img_id'] = sample_id
        dataset.data[sample_id]['image'] = dataset.get_image_path(sample_id)

        # need gt_overlaps as a dense array for argmax
        gt_overlaps = dataset.data[sample_id]['gt_overlaps'].toarray()

        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)

        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        dataset.data[sample_id]['max_classes'] = max_classes
        dataset.data[sample_id]['max_overlaps'] = max_overlaps

        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

    return dataset


def rank_data_ratio(dataset):

    data = dataset.data
    # rank data based on the ratio between width and height.
    ratio_large = 2  # largest ratio to preserve.
    ratio_small = 0.5  # smallest ratio to preserve.

    ratio_list = []
    for sample_id in data.keys():
        width = data[sample_id]['width']
        height = data[sample_id]['height']
        ratio = width / float(height)

        if ratio > ratio_large:
            data[sample_id]['need_crop'] = 1
            ratio = ratio_large
        elif ratio < ratio_small:
            data[sample_id]['need_crop'] = 1
            ratio = ratio_small
        else:
            data[sample_id]['need_crop'] = 0

        ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)

    return ratio_list[ratio_index], ratio_index


def filter_dataset(dataset):
    # todo if dataset written correctly it should never pick up any samples with 0 boxes
    """
    remove images (data samples) that don't have any gt boxes
    :param dataset: non filtered data
    :return: filtered data
    """
    print('before filtering, there are %d images...' % (len(dataset.data)))
    for sample_id in dataset.data.keys():
        if len(dataset.data[sample_id]['boxes']) == 0:
            del dataset.data[sample_id]

    print('after filtering, there are %d images...' % (len(dataset.data)))

    return dataset
