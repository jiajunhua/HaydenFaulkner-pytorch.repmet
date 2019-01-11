"""
DetectionSampler

Returns the batches (in form or sample indexs) in a simple shuffled order, assumes use with DetectionWrapper
dataset where samples are ordered based on img size and split into chunks of size batch_size where all samples
in a batch have same input size and aspect ratio

For example if batch_size=2 and n_samples=5, the DetectionWrapper organises data so aspect ratios are clumped
in batches eg. [.5, .5, .75, .75, .76]

This sampler basically just returns these clumps in a shuffled order by shuffling the starting indexs eg. 0,2

Modified from https://github.com/jwyang/faster-rcnn.pytorch/blob/pytorch-1.0/trainval_net.py
"""

import torch
from torch.utils.data.sampler import Sampler


class DetectionSampler(Sampler):

    def __init__(self, n_samples, batch_size):

        # super(DetectionSampler, self).__init__()

        # set instance variables
        self.n_samples = n_samples
        self.n_per_batch = int(n_samples / batch_size)  # automatically floors, we will just add extras on end if needed
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()

        # Do we need some leftover indexs?
        self.leftover_flag = False
        if n_samples % batch_size:
            self.leftover_inds = torch.arange(self.n_per_batch * batch_size, n_samples).long()
            self.leftover_flag = True

    def __iter__(self):

        # Get the starting indexs for each batch in a shuffled form
        shuffled_start_inds = torch.randperm(self.n_per_batch).view(-1, 1) * self.batch_size
        
        # Fill in the indexs between each starting index
        self.shuffled_inds = shuffled_start_inds.expand(self.n_per_batch, self.batch_size) + self.range

        # Flatten the indexs
        self.shuffled_inds = self.shuffled_inds.view(-1)

        # Append the leftovers
        if self.leftover_flag:
            self.shuffled_inds = torch.cat((self.shuffled_inds, self.leftover_inds), 0)

        # Return an iterable of the shuffled batches
        return iter(self.shuffled_inds)

    def __len__(self):
        return self.n_samples


if __name__ == "__main__":
    # use this for debugging and checks
    from utils.debug import set_working_dir
    from config.config import config
    from data_loading.sets.pascal_voc import PascalVOCDataset
    from data_loading.sets.combined import CombinedDataset
    from data_loading.detection_wrapper import DetectionWrapper

    # set the working directory as appropriate
    set_working_dir()

    # set a batch_size
    batch_size = 10

    # load the dataset
    datasetA = PascalVOCDataset(root_dir=config.dataset.root_dir, split='train', use_flipped=False)
    datasetB = PascalVOCDataset(root_dir=config.dataset.root_dir, split='val', use_flipped=True)
    datasetC = CombinedDataset(datasets=[datasetA, datasetB])
    datasetD = DetectionWrapper(datasetC, training=True, batch_size=batch_size)

    # setup the the sampler
    sampler = DetectionSampler(n_samples=len(datasetD), batch_size=batch_size)

    # setup the dataloader
    dataloader = torch.utils.data.DataLoader(datasetD, batch_size=batch_size, sampler=sampler)

    for batch in iter(dataloader):
        img, im_info, gt_boxes, num_boxes = batch
