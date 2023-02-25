"""抽取和feedback相同的user数绘制CDF
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import json
import re

import utils.MyDrawer as MyDrawer
from utils.MyTrie import Trie

def tmp_prepare_data(in_dir, file_name):
    normal_data_path = os.path.join(in_dir, file_name)
    df = pd.read_csv(normal_data_path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.reset_index()
    df.rename(columns={"index": "unique_id"}, inplace=True)
    return df

def draw_session_pagenum(in_dir, normal_name, feedback_name):
    """绘制每个session的页面数目的分布，使用一周的数据
    注意：下面的代码只对每次读取一天的数据有意义，如果读取多天的数据，需要修改代码
    Args:
        in_dir (str): 输入文件的目录
        normal_name (str): 正常文件的名字
        feedback_name (str): 反馈文件的名字
    Returns:
        None
    """
    normal = tmp_prepare_data(in_dir, normal_name)
    feedback = tmp_prepare_data(in_dir, feedback_name)

    normal_cnt = normal.groupby("session_id")['unique_id'].count()
    feedback_cnt = feedback.groupby("session_id")['unique_id'].count()

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
        xlabel="# of Pages of Session",
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
        "../experiment/figures/sessionnum_frequency.png", bbox_inches='tight')
    json.dump(normal_cnt.values.tolist(), open("../experiment/figures/normal_cnt.json", "w"), indent = 4) # session中的页面长度
    json.dump(feedback_cnt.values.tolist(), open("../experiment/figures/feedback_cnt.json", "w"), indent = 4)

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

def draw_user_sessionnum(in_dir, normal_name, feedback_name):
    """绘制每个user的session数目的分布
    Args:
        in_dir (str): 输入文件夹
        normal_name (str): normal的文件名
        feedback_name (str): feedback的文件名
    Returns:
        None
    """
    normal = tmp_prepare_data(in_dir, normal_name)
    feedback = tmp_prepare_data(in_dir, feedback_name)

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
        xlabel="# of Sessions of a User",
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
        "../experiment/figures/usernum_frequency.png", bbox_inches='tight')
    # json.dump(normal_cnt.values.tolist(), open("../experiment/figures/usernum_normal_cnt.json", "w"))
    # json.dump(feedback_cnt.values.tolist(), open("../experiment/figures/usernum_feedback_cnt.json", "w")) # session中的页面长度

def filter_by_ifurl(url):
    """根据是否是url过滤
    Args:
        url (str): 待过滤的url
    Returns:
        filter_or_not (bool) : 是否被过滤掉，是为True, 否为False
    """
    if re.match(r'^https?:\/\/', url):
        return False
    else:
        return True

def draw_pagenum_distribution(data_path):
    """根据data/page2num-1.csv绘制页面的分布
    Args:
        data_path (str): page2num-1.json的路径
    """
    drawer = MyDrawer.MyDrawer()

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
        "../experiment/figures/pagenum_distribution.png", bbox_inches='tight')
    
    # 多少是url，多少不是url
    pagename_list, pagename_cnt = list(page2num.keys()), list(page2num.values())

    # 根据url本身做过滤（是否是url、结尾的拓展名、是否在lastword_dict中）
    index_list = list(map(filter_by_ifurl, pagename_list))
    print("ratio of is_url: {:2f}%".format(sum(index_list) / len(index_list) * 100)) # 13.28%
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
    g.view(filename = 'Trie', directory = '../experiment/figures/', cleanup = False)

if __name__ == "__main__":
    # in_dir, normal_name, feedback_name = pre_args.in_dir, pre_args.normal_names[0], pre_args.feedback_names[0]
    # draw_session_pagenum(in_dir, normal_name, feedback_name)
    # in_dir, normal_name, feedback_name = pre_args.in_dir, pre_args.normal_names[0], pre_args.feedback_names[0]
    # draw_user_sessionnum(in_dir, normal_name, feedback_name)

    # data_path = "../experiment/page2nums/page2num-1.json"
    # draw_pagenum_distribution(data_path)
    pass
    