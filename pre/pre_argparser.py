"""建立字典的参数
"""
import argparse
import time
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    '-preprocess_root',
    type=str,
    default=os.path.abspath(os.path.dirname(__file__)),
    help="preprocess dir root"
)  # 当前预处理的根目录pre/

# prepreprocess.py的参数
parser.add_argument('-in_dir_prepre', type=str,
                    default="../data/datasets/", help='input file directory') # 输入文件的目录
parser.add_argument('-feedback_names_prepre', nargs='+', default=["feedback.csv"], help='feedback file name') # 反馈文件的名字
parser.add_argument('-normal_names_prepre', nargs='+', default=["normal.csv"], help='normal file name') # 正常文件的名字
parser.add_argument('-output_dir_prepre', type=str, default="../data/prepreprocess/", help='output file name') # 输出的page2num文件夹的名字

# generate_page2num.py的参数
parser.add_argument('-in_dir_gen', type=str,
                    default="../data/datasets/", help='input file directory') # 输入文件的目录
parser.add_argument('-feedback_names_gen', nargs='+', default=["feedback.csv"], help='feedback file name') # 反馈文件的名字
parser.add_argument('-normal_names_gen', nargs='+', default=["normal.csv"], help='normal file name') # 正常文件的名字
parser.add_argument('-output_dir_gen', type=str, default="../data/page2nums/", help='output file name') # 输出的page2num文件夹的名字
parser.add_argument('-page2num_names_gen', nargs='+', default=["page2num-simulate.json"], help='page2num file name') # 输出的page2num文件的名字

# generate_code.py的参数
parser.add_argument('-page2num_dir', type=str, default='../data/page2nums/', help="orgin directory of page2num")  # page2num的原目录
parser.add_argument('-page2num_names', nargs='+', default=['page2num-1.json'], help="orgin filename of page2num")  # page2num的原文件
parser.add_argument('-page2num_merge_name', default='page2num-merge.json', help="filename of page2num after merge")  # 很多个page2num合并后的文件
parser.add_argument('--simulate', action='store_true',
                    default=False, help='if simulate or not')
parser.add_argument('-page2num_afterwash_name', type=str, default='page2num-afterwash.json', help="filename of page2num after wash")  # 预处理后的page2num文件

# preprocess.py的参数
parser.add_argument('-in_dir_pre', type=str,
                    default="../data/datasets/", help='input file directory') # 输入文件的目录
parser.add_argument('-feedback_names_pre', nargs='+', default=["feedback.csv"], help='feedback file name') # 反馈文件的名字
parser.add_argument('-normal_names_pre', nargs='+', default=["normal.csv"], help='normal file name') # 正常文件的名字
parser.add_argument('-output_dir_pre', type=str, default="../data/preprocess/", help='output file name') # 输出的page2num文件夹的名字

pre_args = parser.parse_args()

# 设置时间
cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
pre_args.cur_time = cur_time

# 根据simulate设置page2num_origin_path'，加入data/page2num-simulate.json'
if pre_args.simulate:
    pre_args.page2num_names.append(
        'page2num-simulate.json')

