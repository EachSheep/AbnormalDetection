"""pre文件夹下的参数
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
                    default="/home/hiyoungshen/Source/Archive/ICWS2023/experiment/datasets", help='input file directory') # 输入文件的目录
parser.add_argument('-feedback_names_prepre', nargs='+', default=["feedback.csv"], help='feedback file name') # 反馈文件的名字
parser.add_argument('-normal_names_prepre', nargs='+', default=["normal.csv"], help='normal file name') # 正常文件的名字
parser.add_argument('-output_dir_prepre', type=str, default="../experiment/prepreprocess/", help='output file name') # 输出的page2num文件夹的名字
parser.add_argument('-output_set_prepre', type=str, default="../experiment/assets/", help='output json file name') # 输出的lowercase2uppercase.json存放的目录

# g_page2num.py的参数
parser.add_argument('-in_dir_gen', type=str,
                    default="../experiment/prepreprocess/", help='input file directory') # 输入文件的目录
parser.add_argument('-feedback_names_gen', nargs='+', default=["feedback.csv"], help='feedback file name') # 反馈文件的名字
parser.add_argument('-normal_names_gen', nargs='+', default=["normal.csv"], help='normal file name') # 正常文件的名字
parser.add_argument('-output_dir_gen', type=str, default="../experiment/assets/", help='output file name') # 输出的page2num文件夹的名字
parser.add_argument('-page2num_names_gen', nargs='+', default=["page2num.json"], help='page2num file name') # 输出的page2num文件的名字

# g_lastword_dict.py的参数
parser.add_argument('-page2num_dir', type=str, default='../experiment/assets/', help="orgin directory of page2num")  # page2num的原目录
parser.add_argument('-page2num_names', nargs='+', default=['page2num.json'], help="orgin filename of page2num")  # page2num的原文件
parser.add_argument('-output_dir_lastword', type=str, default="../experiment/assets/", help='output file name') # 输出的lastword_dict文件夹的名字
parser.add_argument('-lastword_dict_names', nargs='+', default=['lastword_dict.json'], help="filename of lastword_dict")  # lastword_dict的文件名称
parser.add_argument('--simulate', action='store_true',
                    default=False, help='if simulate or not')

# g_page_code.py的参数
parser.add_argument('-lastword_dict_dir', type=str, default='../experiment/assets/', help="orgin directory of lastword_dict")  # lastword_dict的原目录
parser.add_argument('-lastword_dict_name', type=str, default='lastword_dict.json', help="orgin filename of lastword_dict")  # lastword_dict的原文件
parser.add_argument('-output_dir_lastword_dict', type=str, default="../experiment/assets/", help='output directory') # 

# preprocess.py的参数
parser.add_argument('-in_dir_pre', type=str,
                    default="../experiment/preprocess/", help='input file directory') # 输入文件的目录
parser.add_argument('-feedback_names_pre', nargs='+', default=["feedback.csv"], help='feedback file name') # 反馈文件的名字
parser.add_argument('-normal_names_pre', nargs='+', default=["normal.csv"], help='normal file name') # 正常文件的名字
parser.add_argument('-output_dir_pre', type=str, default="../experiment/preprocess/", help='output file name') # 输出的page2num文件夹的名字

pre_args = parser.parse_args()

# 设置时间
cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
pre_args.cur_time = cur_time

# 根据simulate设置page2num_origin_path'，加入data/page2num-simulate.json'
if pre_args.simulate:
    pre_args.page2num_names.append(
        'page2num-simulate.json')

