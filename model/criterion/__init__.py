import torch
from model.criterion.deviation_loss import DeviationLoss
from model.criterion.binary_focal_loss import BinaryFocalLoss

def build_criterion(args):
    if args.criterion == "deviation":
        print("Loss : Deviation")
        return DeviationLoss(args)
    elif args.criterion == "BCE":
        print("Loss : Binary Cross Entropy")
        return torch.nn.BCEWithLogitsLoss()
    elif args.criterion == "focal":
        print("Loss : Focal")
        return BinaryFocalLoss()
    else:
        raise NotImplementedError