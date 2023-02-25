import argparse
import time
import torch

parser = argparse.ArgumentParser()

# 当前实验设置
parser.add_argument('-root', type=str,
                    default='./', help="project root")  # 项目根目录
parser.add_argument("-random_seed", type=int, default=42,
                    help="the random seed number")
parser.add_argument('-experiment_dir', type=str, default='./experiment',
                    help="experiment dir root")  # 存放当前实验的参数、模型、日志等
parser.add_argument('-log_dir', type=str, default='log_dir',
                    help="name of log_dir")  # 存放当前实验保存的参数,在experiment_dir下
parser.add_argument('-log_label', type=int, default=1,
                    help="name of log_label")  # 存放当前实验保存的参数,在experiment_dir下,和log_dir组成log_dir-log_label，用于区分不同的实验
parser.add_argument('-weight_name', type=str,
                    default='model-{}.pkl', help="the name of the model weight")
parser.add_argument('-cache_dir', type=str,
                    default='', help="the abs path of the cache dir")

# 训练设置
parser.add_argument('-workers', type=int, default=4,
                    metavar='N', help='dataloader threads')
parser.add_argument('--no_cuda', action='store_true',
                    default=False, help='disables CUDA training')

# 数据集设置
parser.add_argument('-dataset_root', type=str,
                    default='./experiment/preprocess/', help="dataset root")  # 数据集存放的根目录
parser.add_argument('-file_name_abnormal', type=str, default='feedback.csv',
                    help="anomaly file name")  # 异常数据文件名，也就是feedback用户的行为轨迹
parser.add_argument('-file_name_normal', type=str, default='normal.csv',
                    help="normal file name")  # 正常数据文件名，也就是normal用户的行为轨迹
parser.add_argument('--use_cache', action='store_true', default=False,
                    help="user cache or not")  # 是否使用之前使用数据集直接生成的矩阵
parser.add_argument('-vocab_dict_path', type=str, default='experiment/assets/page2idx.json',
                    help="vocab dict path")  # 页面->id的字典存放路径
parser.add_argument('-data_type', type=str, default='pagesession',
                    help="pagesession, pageuser, worduser")  # 按照session切分还是按照user切分
parser.add_argument('-max_seq_len', type=int, default=200,
                    help="vocab dict path")  # 页面->id的字典存放路径
parser.add_argument('-train_ratio', type=float, default=0.8,
                    help="train test split ratio")  # 训练集和测试集划分的比例

# 模型测试设置
parser.add_argument('-test_set', type=str, default="valid",
                    help="use valid set or test set, valid or test")  # 使用训练集中划分出来的测试集测试还是使用新的测试集测试

# 模型设置
parser.add_argument('-backbone', type=str, default='lstma',
                    help="the backbone network")
parser.add_argument('-vocab_size', type=int, default=10000,
                    help="the vocab_size")  # 词汇表大小
parser.add_argument('-embedding_dim', type=int,
                    default=280, help="the embedding_dim")
# lstm, gru专属参数
parser.add_argument('-hidden_dim', type=int,
                    default=200, help="the hidden_dim")
parser.add_argument('-output_dim', type=int, default=1, help="the output_dim")
parser.add_argument('-n_layers', type=int, default=1, help="the n_layers")
parser.add_argument('--use_bidirectional', action='store_true',
                    default=False, help="use bidirectional or not")
parser.add_argument('--use_dropout', action='store_true',
                    default=False, help="use dropout or not")
# transformer专属参数
parser.add_argument('-ffn_num_hiddens', type=int, default=280, help="the ffn_num_hiddens")
parser.add_argument('-num_heads', type=int, default=4, help="the num_heads")
parser.add_argument('-num_layers', type=int, default=2, help="the num_layers")
parser.add_argument('-dropout', type=float, default=0.5, help="the dropout")

# 损失函数和优化器设置
parser.add_argument('-criterion', type=str,
                    default='deviation', help="the loss function")
parser.add_argument('-optimizer', type=str,
                    default='Adam', help="the optimizer")
parser.add_argument('-lr', type=float, default=0.0002, help="lr")
parser.add_argument('-weight_decay', type=float,
                    default=1e-5, help="weight_decay")
parser.add_argument('-scheduler_step_size', type=int,
                    default=10, help="scheduler_step_size")
parser.add_argument('-scheduler_gamma', type=float,
                    default=0.1, help="scheduler_gamma")

# 训练参数设置
parser.add_argument("-epochs", type=int, default=50,
                    help="the number of epochs")
parser.add_argument("-batch_size", type=int, default=48,
                    help="batch size used")
parser.add_argument("-steps_per_epoch", type=int, default=40, # 一个epoch中的batch数
                    help="the number of batches per epoch")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
args.weight_name = args.weight_name.format(cur_time)
args.cur_time = cur_time