
from math import ceil

import numpy as np

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def forward(model, dataset, batch_size):
    """Compute representations for input in chunks."""
    chunks = int(ceil(float(len(dataset)) / batch_size))
    outputs = []
    labels = []
    model.eval()
    loader = DataLoader(dataset,
                        batch_size=batch_size,  #chunks,
                        shuffle=False,  # don't shuffle as we take labels in order in cluster update
                        num_workers=1)

    with torch.no_grad():  # prevents computation graph from being made
        for batch_idx, (inputs, labels_) in enumerate(loader):
            inputs = inputs.to(device)
            output = model(inputs)
            outs = output.data
            outputs.append(outs.cpu().numpy())
            labels.append(labels_.cpu().numpy())
    return np.vstack(outputs), np.hstack(labels)
