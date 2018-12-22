
from math import ceil

import numpy as np

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def forward(model, dataset, chunk_size):
    """Compute representations for input in chunks."""
    chunks = int(ceil(float(len(dataset)) / chunk_size))
    outputs = []
    model.eval()
    trainloader = DataLoader(dataset,
                             batch_size=chunk_size,#chunks,
                             shuffle=False,  # don't shuffle as we take labels in order in cluster update
                             num_workers=1)

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        output = model(inputs)
        outs = output.data
        outputs.append(outs.cpu().numpy())
    return np.vstack(outputs)