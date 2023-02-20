"""
运行命令: python merge_lastword_dict.py -n1 ./train_lastword_dict_path.json -n2 ./test_lastword_dict_path.json -n3 ./lastword_dict1.json
例如: python merge_lastword_dict.py -n1 ./lastword_dict.json -n2 ./lastword_dict.json -n3 ./lastword_dict1.json
"""

import argparse
import time
import json

parser = argparse.ArgumentParser()

# 当前实验设置
parser.add_argument('-n1', type=str,
                    default='./lastword_dict.json', help="name 1")  # 项目根目录
parser.add_argument("-n2", type=str, 
                    default='./lastword_dict.json', help="name 2")
parser.add_argument("-n3", type=str, 
                    default='./lastword_dict1.json', help="name 2")
args = parser.parse_args()

def merge_dict(list1, list2):
    return list(set(list1 + list2))

if __name__ == "__main__":
    list1 = json.load(open(args.n1, 'r'))
    list2 = json.load(open(args.n2, 'r'))
    list3 = merge_dict(list1, list2)
    with open(args.n3, 'w') as f:
        json.dump(list3, f, indent=4)