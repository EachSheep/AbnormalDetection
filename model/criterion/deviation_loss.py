import torch
import torch.nn as nn


class DeviationLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, y_pred, y_true):
        confidence_margin = 5.
        # size=5000 is the setting of l in algorithm 1 in the paper
        if self.args.cuda:
            ref = torch.normal(mean=0., std=torch.full([5000], 1.)).cuda()
        else:
            ref = torch.normal(mean=0., std=torch.full([5000], 1.))
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs((confidence_margin - dev).clamp_(min=0.))
        return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)

        # 这里的均值和方差也许可以学
