import argparse
import os
import time

import torch

from Trainer import Trainer

parser = argparse.ArgumentParser()

# 当前实验设置
parser.add_argument("--random_seed", type=int, default=42,
                    help="the random seed number")
parser.add_argument('--experiment_dir', type=str, default='./experiment',
                    help="experiment dir root")  # 存放当前实验的参数、模型、日志等
parser.add_argument('--weight_name', type=str,
                    default='model-{}.pkl', help="the name of the model weight")

# 训练设置
parser.add_argument('--workers', type=int, default=4,
                    metavar='N', help='dataloader threads')
parser.add_argument('--no_cuda', action='store_true',
                    default=False, help='disables CUDA training')

# 数据集设置
parser.add_argument('--dataset_root', type=str,
                    default='./data/', help="dataset root")  # 数据集存放的根目录
parser.add_argument('--file_name_abnormal', type=str, default='feedback.csv',
                    help="anomaly file name")  # 异常数据文件名，也就是feedback用户的行为轨迹
parser.add_argument('--file_name_normal', type=str, default='normal.csv',
                    help="normal file name")  # 正常数据文件名，也就是normal用户的行为轨迹
parser.add_argument('--filter_num', type=float, default=10, help="filter num")  # 筛除用户行为轨迹中出现次数少于filter_num的session
parser.add_argument('--train_ratio', type=float, default=0.8,
                    help="train test split ratio")  # 训练集和测试集划分的比例

# 模型设置
parser.add_argument('--backbone', type=str, default='lstma',
                    help="the backbone network")
parser.add_argument('--vocab_size', type=int, default=1000,
                    help="the vocab_size")  # 词汇表大小
parser.add_argument('--embedding_dim', type=int,
                    default=300, help="the embedding_dim")
parser.add_argument('--hidden_dim', type=int,
                    default=512, help="the hidden_dim")
parser.add_argument('--output_dim', type=int, default=1, help="the output_dim")
parser.add_argument('--n_layers', type=int, default=1, help="the n_layers")
parser.add_argument('--use_bidirectional', type=bool,
                    default=False, help="use bidirectional or not")
parser.add_argument('--use_dropout', type=bool,
                    default=False, help="use dropout or not")

# 损失函数和优化器设置
parser.add_argument('--criterion', type=str,
                    default='deviation', help="the loss function")
parser.add_argument('--optimizer', type=str,
                    default='Adam', help="the optimizer")
parser.add_argument('--lr', type=float, default=0.0002, help="lr")
parser.add_argument('--weight_decay', type=float,
                    default=1e-5, help="weight_decay")
parser.add_argument('--scheduler_step_size', type=int,
                    default=10, help="scheduler_step_size")
parser.add_argument('--scheduler_gamma', type=float,
                    default=0.1, help="scheduler_gamma")

# 训练参数设置
parser.add_argument("--epochs", type=int, default=50,
                    help="the number of epochs")
parser.add_argument("--batch_size", type=int, default=48,
                    help="batch size used in SGD")
parser.add_argument("--steps_per_epoch", type=int, default=20,
                    help="the number of batches per epoch")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
args.weight_name = args.weight_name.format(cur_time)

if __name__ == '__main__':

    torch.manual_seed(args.random_seed)

    trainer = Trainer(args)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    args_dict = args.__dict__
    setting_path = os.path.join(args.experiment_dir, 'setting.txt')
    with open(setting_path, 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for cur_arg, cur_value in args_dict.items():
            f.writelines(cur_arg + ' : ' + str(cur_value) + '\n')
        f.writelines('------------------- end -------------------')

    roc_list, ap_list = [], []
    for cur_epoch in range(0, trainer.args.epochs):
        trainer.train(cur_epoch)
        if (cur_epoch + 1) % 10 == 0:
            roc, ap = trainer.eval()
            roc_list.append(roc)
            ap_list.append(ap)
            print('cur_epoch: ', cur_epoch, 'roc: ', roc, 'ap: ', ap)