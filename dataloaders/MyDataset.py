from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    """载入自定义数据集
    Args:
        args: 参数
        data (n, m): 数据
        label (n,): 标签
        train: 是否为训练集, True代表载入训练集, False代表载入测试集
    """

    def __init__(self, args, data, label, **kwargs):
        super(MyDataset).__init__()
        self.args = args
        self.data = data # 一个向量，对应一个标签
        self.label = label
        self.kwargs = kwargs # 包含每个数据的长度（len），每个数据的session_id（sid），每个数据的user_id（uid）

        self.idx = torch.arange(len(self.data))
        self.normal_idx = torch.argwhere(self.label == 0).flatten()
        self.outlier_idx = torch.argwhere(self.label == 1).flatten()

        self.valid_lens = self.kwargs['valid_lens']

    def __len__(self):
        return len(self.data)

    def normal_num(self):
        return len(self.normal_idx)
    
    def abnormal_num(self):
        return len(self.outlier_idx)

    def __getitem__(self, index):
        sample = {'data': self.data[index], 'label': self.label[index], 'valid_lens': self.valid_lens[index]}
        return sample