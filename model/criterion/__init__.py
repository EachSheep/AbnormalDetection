import torch
from model.criterion.deviation_loss import DeviationLoss
from model.criterion.binary_focal_loss import BinaryFocalLoss
import torch.nn as nn
import torch.nn.functional as F

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight # 二分类中正负样本的权重，第一项为正类权重，第二项为负类权重

    def forward(self, input, target):
        # 根据target是1还是0，将input中对应的值乘以相应的权重
        weight = torch.zeros_like(input)
        weight[target == 1] = self.weight[0]
        weight[target == 0] = self.weight[1]
        return F.binary_cross_entropy_with_logits(input, target,
                                                  weight)

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveWithLogitsLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.weighted_BCE_with_logits_loss = WeightedBCEWithLogitsLoss(args.weight)
        self.sup_con_loss = SupConLoss()

    def forward(self, X, input, target):
        # X: [n, c]
        X = X.view(X.shape[0], 1, -1)
        return self.weighted_BCE_with_logits_loss(input, target) + \
            self.sup_con_loss(self, X, labels=target)
            

def build_criterion(args):
    if args.criterion == "deviation":
        print("Loss : Deviation")
        return DeviationLoss(args)
    elif args.criterion == "BCE":
        print("Loss : Binary Cross Entropy")
        return torch.nn.BCEWithLogitsLoss()
    elif args.criterion == "WeightedBCE":
        print("Loss : Weighted Binary Cross Entropy")
        return WeightedBCEWithLogitsLoss(args.weight)
    elif args.criterion == "Contrastive":
        print("Loss : Contrastive")
        return ContrastiveWithLogitsLoss(args)
    elif args.criterion == "focal":
        print("Loss : Focal")
        return BinaryFocalLoss()
    else:
        raise NotImplementedError