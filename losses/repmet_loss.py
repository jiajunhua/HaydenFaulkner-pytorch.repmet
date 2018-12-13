import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from utils.functions import make_one_hot


class RepmetLoss(nn.Module):

    def __init__(self, alpha=1.0):
        super(RepmetLoss, self).__init__()
        self.alpha = alpha

    def forward(self, distances, labels, alpha=1.0):
        """
        Equation (4) of repmet paper

        :param dists: n_samples x n_classes x n_k
        :param labels: n_samples
        :param alpha:
        :return:
        """

        self.alpha = alpha

        self.n_samples = distances.size()[0]
        self.n_classes = distances.size()[1]
        self.n_k = distances.size()[2]

        # make mask with ones where correct class, zeros otherwise
        mask = make_one_hot(labels, n_classes=self.n_classes)
        mask_cor = mask.transpose(0, 1).repeat(1, self.n_k).view(-1, self.n_samples).transpose(0, 1)
        mask_inc = ~mask_cor

        valmax, argmax = distances.max(-1)
        valmax, argmax = valmax.max(-1)

        print(mask_inc.shape)
        print(valmax.shape)
        # print(valmax.view(5,self.n_classes*self.n_k).shape)
        # t = mask_inc*valmax

        min_cor, _ = (distances + (valmax*mask_inc)).min(1)
        min_inc, _ = (distances + (mask_cor*valmax)).min(1)

        losses = F.relu(min_cor - min_inc + self.alpha)

        total_loss = torch.mean(losses)

        return total_loss, losses

# class RepmetLossFunction(Function):
#
#     @staticmethod
#     def forward(ctx, input, weight, distances, labels, alpha=1.0):
#         """
#         Equation (4) of repmet paper
#
#         :param dists: n_samples x n_classes x n_k
#         :param labels: n_samples
#         :param alpha:
#         :return:
#         """
#
#         n_samples = distances.size()[0]
#         n_classes = distances.size()[1]
#         n_k = distances.size()[2]
#
#         # make mask with ones where correct class, zeros otherwise
#         mask = make_one_hot(labels, n_classes=n_classes)
#         mask_cor = mask.transpose(0, 1).repeat(1, n_k).view(-1, n_samples).transpose(0, 1)
#         mask_inc = ~mask_cor
#
#         valmax, argmax = distances.max(1)
#
#         min_cor, _ = (distances + (mask_inc*valmax)).min(1)
#         min_inc, _ = (distances + (mask_cor*valmax)).min(1)
#
#         losses = F.relu(min_cor - min_inc + alpha)
#
#         total_loss = torch.mean(losses)
#
#         return total_loss, losses
#
#     @staticmethod
#     def backward(ctx, grad_output):

if __name__ == "__main__":
    print("Simple test of emb loss")
    repmet = RepmetLoss()

    p = [[1, 0, 0],
         [1, 0, 0],
         [1, 0, 0],
         [.25, .5, .25],
         [.75, 0, .25]]
    l = [1, 0, 0, 1, 2]
    d = [[[1, 0.00], [1, 0.00], [1, 0.00]], #C0K0, C0K1, C1K0 ... C2K1
         [[0.001, 0.001], [1, 1], [1, 1]],
         [[0.001, 0.001], [1, 1], [1, 1]],
         [[0.001, 0.002], [1, 1], [1, 1]],
         [[.6, 1], [.6, 1], [.5, 0.001]]]

    d = torch.autograd.Variable(torch.Tensor(d), requires_grad=True)
    l = torch.autograd.Variable(torch.Tensor(l), requires_grad=True)
    loss = repmet(d, l)

    print('done')