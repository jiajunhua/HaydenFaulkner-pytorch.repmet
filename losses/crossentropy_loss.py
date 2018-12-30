import torch

from torch.nn import functional as F
from torch.nn.modules import Module


class CrossEntropyLoss(Module):
    """
    Loss class for cross entropy

    Using this instead of nn.CrossEntropyLoss because we calc acc and pred inside rather than outside the loss functions
    """
    def __init__(self, softmaxed=True, weight=None):
        super(CrossEntropyLoss, self).__init__()
        self.softmaxed = softmaxed
        self.weight = weight

    def forward(self, input, target):
        # we are expecting the outputs to be one per class
        assert input.shape[1] >= target.max().item(), "%d" % target.max().item()

        if not self.softmaxed:
            input = F.log_softmax(input, dim=1)

        if self.weight is not None:
            losses = F.nll_loss(input, target, weight=self.weight, reduction='none')
            total_loss = torch.mean(losses)
            _, pred = input.max(1)
            acc = pred.eq(target.squeeze()).float().mean()
        else:
            losses = F.nll_loss(input, target, reduction='none')
            total_loss = torch.mean(losses)
            _, pred = input.max(1)
            acc = pred.eq(target.squeeze()).float().mean()

        return total_loss, losses, pred, acc
