"""首先将原始数据处理成我们期望的表的形式保存下来
"""
import os
import pandas as pd
import json

from pre_argparser import pre_args
from wash_pagename import *
from PreProcesser import preprocess


def tmp_prepare_data(in_dir, file_name):
    data_path = os.path.join(in_dir, file_name)
    df = pd.read_csv(data_path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.reset_index()
    df.rename(columns={"index": "unique_id"}, inplace=True)
    return df


def prepreprocess(in_dir, normal_names, feedback_names, output_dir, **kwargs):
    """输入原文件，输出预处理后的文件和page2num文件
    Args:
        in_dir: 输入文件的目录
        normal_names: 正常文件的名字
        feedback_names: 反馈文件的名字
        output_dir: 输出的page2num文件所在目录
        **kwargs: 其他参数
    Returns:
        None
    """
    min_seq_len = kwargs["min_seq_len"]
    page2num_names = kwargs["page2num_names"]

    for normal_name, feedback_name, page2num_name in zip(normal_names, feedback_names, page2num_names):

        normal = tmp_prepare_data(in_dir, normal_name)
        print('正常用户：根据session中的页面筛选前用户的轨迹数为：', len(normal))
        feedback = tmp_prepare_data(in_dir, feedback_name)
        print('异常用户：根据session中的页面筛选前用户的轨迹数为：', len(feedback))

        # 去除重复的session_id
        feedback_session = feedback["session_id"].unique()
        normal = normal[~normal["session_id"].isin(feedback_session)]
        normal_session = normal["session_id"].unique()
        feedback = feedback[~feedback["session_id"].isin(normal_session)]

        # # 去除重复的user_id
        # feedback_user = feedback["user_id"].unique()
        # normal = normal[~normal["user_id"].isin(feedback_user)]
        # normal_user = normal["user_id"].unique()
        # feedback = feedback[feedback["user_id"].isin(normal_user)]

        all = pd.concat([normal, feedback], axis=0).drop(columns=['unique_id']).reset_index(
            drop=True).reset_index().rename(columns={"index": "unique_id"})

        all_page_num = all.groupby('page_name')[
            'unique_id'].count().sort_values(ascending=False)
        page2num = dict(
            zip(all_page_num.index, [int(value) for value in all_page_num.values]))
        json.dump(page2num, open(
            os.path.join(output_dir, page2num_name), "w"), indent=4)

        # 去除nan, 预处理, 筛选过少轨迹的用户
        def tmp(normal):
            normal = normal.dropna()

            normal['page_name'] = normal['page_name'].map(preprocess)
            normal['is_ok'] = normal['page_name'].map(filter_by_url)
            normal['is_ok'] = normal['is_ok'] + \
                normal['page_name'].map(filter_by_freq_url)
            df_nomral_no_process = normal[normal['is_ok'] == True]
            df_normal_need_process = normal[normal['is_ok'] == False]
            df_normal_need_process.loc[df_normal_need_process.index, ['page_name']] = df_normal_need_process['page_name'].map(process_by_force)
            normal = pd.concat([df_nomral_no_process, df_normal_need_process])
            normal = normal.drop(columns=['is_ok'])

            df_normal_cnt = normal.groupby("session_id")['unique_id'].count()
            df_normal_sel_id = df_normal_cnt[df_normal_cnt >= min_seq_len]
            normal = normal[normal['session_id'].isin(
                df_normal_sel_id.index)]
            return normal

        normal = tmp(normal)
        feedback = tmp(feedback)
        normal.to_csv(os.path.join(output_dir, normal_name), index=False)
        feedback.to_csv(os.path.join(output_dir, feedback_name), index=False)


if __name__ == "__main__":

    in_dir = pre_args.in_dir
    normal_names = pre_args.normal_names
    feedback_names = pre_args.feedback_names
    output_dir = pre_args.output_dir
    page2num_names = pre_args.page2num_names_o
    kwargs = {
        "min_seq_len": 10,
        "page2num_names" : page2num_names
    }
    prepreprocess(in_dir, normal_names, feedback_names, output_dir, **kwargs)