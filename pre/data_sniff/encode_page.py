"""使用feedback的用户log，给所有页面进行编码，同时获取session中最大页面的长度
"""
import os
import pandas as pd
import argparse
import time
import sys
import json

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

def encode_page():
    """使用feedback和normal的用户log，给所有页面进行编码，同时获取session中最大页面的长度，使用一周的数据
    """
    normal = tmp_prepare_data("normal.csv")
    feedback = tmp_prepare_data("feedback.csv")
    all = pd.concat([normal, feedback], axis=0).drop(columns=['unique_id']).reset_index(drop=True).reset_index().rename(columns={"index": "unique_id"})

    all_page = list(all.groupby('page_name')['unique_id'].count().sort_values(ascending=False).index)

    page2idx = {
        '<eos>' : 0,
        '<unk>' : 1,  # 未知页面
        '<pad>' : 2
    }
    page2idx.update(dict(zip(all_page, range(3, len(all_page) + 3))))
    json.dump(page2idx, open(f"pre/data/page2idx-{cur_time}.json", "w"), indent=4)

    idx2page = dict(range(len(all_page) + 3), zip(['<eos>', '<unk>', '<pad>'] + list(all_page)))
    json.dump(idx2page, open(f"pre/data/idx2page-{cur_time}.json", "w"), indent=4)

if __name__ == "__main__":
    encode_page()
    pass
    