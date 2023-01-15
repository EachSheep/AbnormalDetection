import torch

pred = torch.tensor([[-0.5816, -0.3873, -1.0215, -1.0145,  0.4053],
        [ 0.7265,  1.4164,  1.3443,  1.2035,  1.8823],
        [-0.4451,  0.1673,  1.2590, -2.0757,  1.7255],
        [ 0.2021,  0.3041,  0.1383,  0.3849, -1.6311]])

values, indices = pred.topk(3, dim=0, largest=True, sorted=True)
print(values)