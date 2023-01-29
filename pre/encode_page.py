"""使用feedback的用户log，给所有页面进行编码，同时获取session中最大页面的长度
"""
import os
import pandas as pd
import argparse
import time
import sys
import json

from wash_pagename import *

cur_login_user = os.getlogin()
cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
cur_abs_working_directory = os.path.abspath(
    "/home/{}/Source/deviation-network-fliggy/".format(cur_login_user))  # 设置当前项目的工作目录
os.chdir(cur_abs_working_directory)
print("current working directory:", os.getcwd())
cur_time = '2023-01-20-21-57-52'

sys.path.append("./")

parser = argparse.ArgumentParser(description='')
parser.add_argument('-infile_directory', type=str,
                    default="data/datasets/", help='input file directory')
args = parser.parse_args()

def tmp_prepare_data(file_name: str):
    normal_data_path = os.path.join(args.infile_directory, file_name)
    df = pd.read_csv(normal_data_path)
    df["date_time"] = pd.to_datetime(df.date_time)
    df = df.reset_index()
    df.rename(columns={"index": "unique_id"}, inplace=True)
    return df

def generate_pagename_cnt():
    """使用feedback和normal的用户log，计算每一个page_name的数量
    """
    normal = tmp_prepare_data("normal.csv")
    feedback = tmp_prepare_data("feedback.csv")
    all = pd.concat([normal, feedback], axis=0).drop(columns=['unique_id']).reset_index(drop=True).reset_index().rename(columns={"index": "unique_id"})

    all_page_num = all.groupby('page_name')['unique_id'].count().sort_values(ascending=False)
    page2num = dict(zip(all_page_num.index, [int(value) for value in all_page_num.values]))
    json.dump(page2num, open(f"pre/data/page2num-{cur_time}.json", "w"), indent=4)

    

def encode_page(pagename_cnt_path : str):
    """使用清洗过的page_name文件，给所有页面进行编码
    Args:
        pagename_cnt_path (str): 清洗过的page_name文件
    Returns:
        page2idx (dict): page_name到idx的映射
        idx2page (dict): idx到page_name的映射
    """
    all_page_num = json.load(open(pagename_cnt_path, 'r'))
    all_page = list(all_page_num.keys())
    page2idx = {
        '<eos>' : 0,
        '<unk>' : 1,  # 未知页面
        '<pad>' : 2
    }
    page2idx.update(dict(zip(all_page, range(3, len(all_page) + 3))))
    json.dump(page2idx, open(f"pre/data/page2idx-{cur_time}.json", "w"), indent=4)
    idx2page = dict(zip(range(len(all_page) + 3), ['<eos>', '<unk>', '<pad>'] + list(all_page)))
    json.dump(idx2page, open(f"pre/data/idx2page-{cur_time}.json", "w"), indent=4)

if __name__ == "__main__":
    # generate_pagename_cnt()

    page2num = json.load(open('pre/data/page2num-origin-2023-01-20-21-57-52.json', 'r'))
    safe_list, safe_num = [], []
    url_list, url_num = list(page2num.keys()), list(page2num.values())

    safe_list, safe_num, url_list, url_num = filter_by_ifurl(safe_list, safe_num, url_list, url_num)
    url_list, url_num = url_preprocess(url_list, url_num)
    safe_list, safe_num, url_list, url_num = filter_by_extension(safe_list, safe_num, url_list, url_num) # 通用的清洗
    safe_list, safe_num, url_list, url_num = filter_by_equal(safe_list, safe_num, url_list, url_num) # 通用的清洗
    safe_list, safe_num, url_list, url_num = filter_by_dict(safe_list, safe_num, url_list, url_num) # 通用的清洗
    safe_list, safe_num, url_list, url_num = filter_by_cnt(safe_list, safe_num, url_list, url_num) # 这个操作放后面，cnt清洗可能出错
    safe_list, safe_num, url_list, url_num = filter_by_special(safe_list, safe_num, url_list, url_num) # 这个操作放后面，特殊情况特殊处理

    safe_dict = dict(zip(safe_list, safe_num))
    json.dump(safe_dict, open('pre/data/safe_dict.json', 'w'), indent=4)
    url_dict = dict(zip(url_list, url_num))
    json.dump(url_dict, open('pre/data/url_dict.json', 'w'), indent=4)

    encode_page(pagename_cnt_path = 'pre/data/safe_dict.json')
    
