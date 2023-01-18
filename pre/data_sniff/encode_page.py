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
    """使用feedback的用户log，给所有页面进行编码，同时获取session中最大页面的长度
    """
    feedback = tmp_prepare_data("feedback.csv")
    feedback_page = feedback['page_name'].unique()
    page2id = dict(zip(feedback_page, range(len(feedback_page))))
    page2id['<eos>'] = len(feedback_page)
    page2id['<pad>'] = len(feedback_page) + 1
    page2id['<unk>'] = len(feedback_page) + 2  # 未知页面
    json.dump(page2id, open(f"pre/data/page2id-{cur_time}.json", "w"))

    session_maxlen = feedback.groupby("session_id")['unique_id'].count().max()
    print("session_maxlen:", session_maxlen)

if __name__ == "__main__":
    encode_page()
    pass
    