"""抽取和feedback相同的user数绘制CDF
"""
import os
import matplotlib.pyplot as plt
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

import pre.utils.MyDrawer as MyDrawer
from pre.utils.MyTrie import Trie

parser = argparse.ArgumentParser(description='data_sniff')
parser.add_argument('-infile_directory', type=str,
                    default="data/datasets/", help='input file directory')
args = parser.parse_args()

def tmp_prepare_data(file_name: str):
    normal_data_path = os.path.join(args.infile_directory, file_name)
    df = pd.read_csv(normal_data_path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.reset_index()
    df.rename(columns={"index": "unique_id"}, inplace=True)

    return df

def draw_session_pagenum():
    """绘制每个session的页面数目的分布，使用一周的数据
    注意：下面的代码只对每次读取一天的数据有意义，如果读取多天的数据，需要修改代码
    """
    normal = tmp_prepare_data("normal.csv")
    feedback = tmp_prepare_data("feedback.csv")
    # all = pd.concat([normal, feedback], axis=0).drop(columns=['unique_id']).reset_index(drop=True).reset_index().rename(columns={"index": "unique_id"})

    # 去除normal中session_id的交集
    normal = normal[~normal["session_id"].isin(feedback["session_id"].unique())]
    # 去除feedback中session_id的交集
    feedback = feedback[~feedback["session_id"].isin(normal["session_id"].unique())]

    normal_cnt = normal.groupby("session_id")['unique_id'].count()
    feedback_cnt = feedback.groupby("session_id")['unique_id'].count()
    # all_cnt = all.groupby("session_id")['unique_id'].count()

    drawer = MyDrawer.MyDrawer()

    cdfvalues_list = [
        normal_cnt.values,
        feedback_cnt.values,
        # all_cnt.values,
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    drawer.drawMergeCDF(
        cdfvalues_list,
        fig_type="frequency",
        xlabel="# of Pages of Session",
        ylabel="CDF",
        color_list=['red', 'blue', 'purple'],
        marker_list=['x', 'x', 'x', 'x'],
        legend_label_list=['normal', 'feedback', 'all'],
        percentagey=True,
        reverse=False,
        xscale="log",
        fig=fig,
        ax=ax
    )
    ax.grid(True)
    plt.savefig(
        "pre/figures/sessionnum_frequency-{}.png".format(cur_time), bbox_inches='tight')
    json.dump(list(normal_cnt.values), open("pre/figures/normal_cnt-{}.json".format(cur_time), "w")) # session中的页面长度
    json.dump(list(feedback_cnt.values), open("pre/figures/feedback_cnt-{}.json".format(cur_time), "w")) # session中的页面长度
    # json.dump(list(all_cnt.values), open("pre/figures/all_cnt-{}.json".format(cur_time), "w")) # session中的页面长度

    def tmp_function(normal, feedback):
        """做一些简单的数据统计
        """
        # session num
        normal_session_num = normal["session_id"].unique()
        print("normal_session_num:", len(normal_session_num))
        feedback_session_num = feedback["session_id"].unique()
        print("feedback_session_num:", len(feedback_session_num))
        merged_session_num = list(set(normal_session_num) | set(feedback_session_num))
        print("merged_session_num:", len(merged_session_num))

        # 看一下nomral数据里的user有多少, 看一下feedback数据里的user有多少，确定一下有多少user是有log的
        normal_user_num = normal["user_id"].unique()
        print("normal_user_num:", len(normal_user_num))
        feedback_user_num = feedback["user_id"].unique()
        print("feedback_user_num:", len(feedback_user_num))
        merged_user_num = list(set(normal_user_num) | set(feedback_user_num))
        print("merged_user_num:", len(merged_user_num))

        # 统计一下不同的页面的个数
        normal_page_num = normal["page_name"].unique()
        print("normal_page_num:", len(normal_page_num))
        feedback_page_num = feedback["page_name"].unique()
        print("feedback_page_num:", len(feedback_page_num))
        merged_page_num = list(set(normal_page_num) | set(feedback_page_num))
        print("merged_page_num:", len(merged_page_num))

        # 统计一下页面的个数
        print("normal_page_num:", len(normal))
        print("feedback_page_num:", len(feedback))
        print("merged_page_num:", len(normal) + len(feedback))

    normal['date'] = normal['date_time'].dt.date
    feedback['date'] = feedback['date_time'].dt.date
    tmp_function(normal, feedback)

    # 第一天
    normal_first = normal[normal['date'] == pd.to_datetime('2023-01-02')]
    feedback_first = feedback[feedback['date'] == pd.to_datetime('2023-01-02')]
    tmp_function(normal_first, feedback_first)

    # # 前六天
    # normal_six = normal[normal['date'] != pd.to_datetime('2023-01-08')]
    # feedback_six = feedback[feedback['date'] != pd.to_datetime('2023-01-08')]
    # tmp_function(normal_six, feedback_six)

    # # 第七天
    # normal_seventh = normal[normal['date'] == pd.to_datetime('2023-01-08')]
    # feedback_seventh = feedback[feedback['date'] == pd.to_datetime('2023-01-08')]
    # tmp_function(normal_seventh, feedback_seventh)

def draw_user_sessionnum():
    """绘制每个user的session数目的分布
    """
    normal = tmp_prepare_data("normal.csv")
    feedback = tmp_prepare_data("feedback.csv")

    normal = normal[~normal["session_id"].isin(feedback["session_id"].unique())] # 去除normal中session_id的交集
    feedback = feedback[~feedback["session_id"].isin(normal["session_id"].unique())] # 去除feedback中session_id的交集

    # 对normal中的session_id进行去重
    normal = normal.drop_duplicates(subset=['session_id'], keep='first')
    feedback = feedback.drop_duplicates(subset=['session_id'], keep='first')

    normal_cnt = normal.groupby("user_id")['session_id'].count()
    feedback_cnt = feedback.groupby("user_id")['session_id'].count()

    drawer = MyDrawer.MyDrawer()

    cdfvalues_list = [
        normal_cnt.values,
        feedback_cnt.values,
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    drawer.drawMergeCDF(
        cdfvalues_list,
        fig_type="frequency",
        xlabel="# of Sessions of User",
        ylabel="CDF",
        color_list=['red', 'blue'],
        marker_list=['x', 'x'],
        legend_label_list=['normal', 'feedback'],
        percentagey=True,
        reverse=False,
        xscale="log",
        fig=fig,
        ax=ax
    )
    ax.grid(True)
    plt.savefig(
        f"pre/figures/usernum_frequency-{cur_time}.png", bbox_inches='tight')
    json.dump(normal_cnt.values.tolist(), open(f"pre/figures/usernum_normal_cnt-{cur_time}.json", "w")) # session中的页面长度
    json.dump(feedback_cnt.values.tolist(), open(f"pre/figures/usernum_feedback_cnt-{cur_time}.json", "w")) # session中的页面长度

def draw_pagenum_distribution():
    """根据data/page2num-origin.csv绘制页面的分布
    """
    drawer = MyDrawer.MyDrawer()

    data_path = "pre/data/page2num-origin-2023-01-20-21-57-52.json"
    page2num = json.load(open(data_path, "r"))
    page2num_list = sorted(page2num.items(), key=lambda x: x[1], reverse=True)

    x_list = [[i for i in range(len(page2num_list))]]
    y_list = [[cur[1] for cur in page2num_list]]
    fig = plt.figure(111)
    ax = fig.add_subplot(111)
    drawer.drawMultipleLineChart(
        x_list,
        y_list,
        xlabel="Page id",
        ylabel="Page num",
        xscale="log",
        yscale="linear",
        color_list=None,
        marker_list=None,
        legend_label_list=None,
        fig=fig,
        ax=ax,
    )
    plt.savefig(
        f"pre/figures/pagenum_distribution-{cur_time}.png", bbox_inches='tight')
    
    # 多少是url，多少不是url
    from wash_pagename import preprocess, merge_url, filter_by_ifurl
    pagename_list, pagename_cnt = list(page2num.keys()), list(page2num.values())
    pagename_list = list(map(preprocess, pagename_list))
    # 经过预处理一些url可能已经重合，进行合并
    pagename_list, pagename_cnt = merge_url(pagename_list, pagename_cnt)

    # 根据url本身做过滤（是否是url、结尾的拓展名、是否在lastword_dict中）
    index_list = list(map(filter_by_ifurl, pagename_list))
    print("ratio of is_url: {:2f}%".format(sum(index_list) / len(index_list) * 100)) # 12.61%
    url_list = [pagename_list[i] for i in range(len(index_list)) if index_list[i]]
    url_cnt = [pagename_cnt[i] for i in range(len(index_list)) if index_list[i]]
    print("ratio of is_url in dataset: {:2f}%".format(sum(url_cnt) / sum(pagename_cnt) * 100)) # 84.54%

    # 对url按照/进行分割，建立字典树，绘制字典树的图
    trie = Trie()
    for i in range(len(url_list)):
        cur_split = url_list[i].split("/")
        cur_split = [cur for cur in cur_split if cur != ""] # 去除空字符串
        trie.insert_multi(cur_split, [url_cnt[i]] * len(cur_split))
    g = trie.draw_trie()
    g.view(filename = 'Trie', directory = 'pre/figures/', cleanup = False)

if __name__ == "__main__":
    # draw_session_pagenum()
    draw_user_sessionnum()
    # draw_pagenum_distribution()
    pass
    