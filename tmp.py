import torch

import torch.nn.functional as F

a = torch.tensor([1, 2, 3])
b = torch.tensor([1, 2, 3])

euclidean_distance = F.pairwise_distance(a, b)

print(euclidean_distance)