import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152



class ResNetEncoder(nn.Module):
    def __init__(self, emb_dim, type=18, fc_dim=None, norm=True, pretrained=True, lock=False):
        super(ResNetEncoder, self).__init__()

        self.fc_dim = fc_dim
        self.norm = norm
        self.pretrained = pretrained

        if type == 18:
            if self.pretrained:
                self.backbone = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
                if lock:
                    for param in self.backbone.parameters():
                        param.requires_grad = False
                ll_size = 512
            elif fc_dim:
                self.backbone = resnet18(pretrained=False, num_classes=fc_dim)
            else:
                self.backbone = resnet18(pretrained=False, num_classes=emb_dim)
        elif type == 50:
            if self.pretrained:
                self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-1])
                if lock:
                    for param in self.backbone.parameters():
                        param.requires_grad = False
                ll_size = 2048
            elif fc_dim:
                self.backbone = resnet50(pretrained=False, num_classes=fc_dim)
            else:
                self.backbone = resnet50(pretrained=False, num_classes=emb_dim)

        if self.pretrained:

            if fc_dim:
                self.fc1 = nn.Linear(ll_size, fc_dim)
                self.bn1 = nn.BatchNorm1d(fc_dim)
                self.fc2 = nn.Linear(fc_dim, emb_dim)
            else:
                self.fc1 = nn.Linear(ll_size, emb_dim)
        else:
            if fc_dim:
                self.bn1 = nn.BatchNorm1d(fc_dim)
                self.fc1 = nn.Linear(fc_dim, emb_dim)

    def forward(self, x):
        if len(x.shape) < 4:
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.backbone(x)

        if self.pretrained:
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            if self.fc_dim:
                x = F.relu(self.bn1(x))
                x = self.fc2(x)
        elif self.fc_dim:
            x = F.relu(self.bn1(x))
            x = self.fc1(x)

        if self.norm:
            x = F.normalize(x)

        return x