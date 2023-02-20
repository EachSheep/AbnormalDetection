"""首先将原始数据处理成我们期望的表的形式保存下来
"""
import os
import pandas as pd
import re
import json

from pre_argparser import pre_args


def tmp_prepare_data(in_dir, file_name):
    """载入原始数据，返回一个dataframe
    Args:
        in_dir (str): 输入文件的目录
        file_name (str): 输入文件的名字
    Returns:
        df (dataframe): 一个dataframe
    """
    data_path = os.path.join(in_dir, file_name)
    df = pd.read_csv(data_path)
    df = df.reset_index()
    df.rename(columns={"index": "unique_id"}, inplace=True)
    return df

lowercase2uppercase = {}

def url_prepreprocess(url):
    """预处理
    Args:
        url (str): 待处理的url
    Returns:
        url (str): 处理后的url
    # 筛除局域网内广播信息
    # index_list = [True if not re.match(r'^https?:\/\/(192\.168|10|172\.1[6-9]|172\.2[0-9]|172\.3[0-1])\.', url) else False for url in url_list]
    """
    url = re.split(r'[^a-zA-Z]', url)
    url = [cur for cur in url if cur]
    global lowercase2uppercase
    [lowercase2uppercase.setdefault(cur.lower(), cur) for cur in url if cur != cur.lower()]
    return url


def g_lowercase2uppercase(in_dir, normal_names, feedback_names, output_dir, **kwargs):
    """输入原文件，输出预预处理后的文件
    Args:
        in_dir: 输入文件的目录
        normal_names: 正常文件的名字
        feedback_names: 反馈文件的名字
        output_dir: 输出的文件所在目录
        **kwargs: 其他参数
    Returns:
        None
    """
    for normal_name, feedback_name in zip(normal_names, feedback_names):

        normal = tmp_prepare_data(in_dir, normal_name)
        feedback = tmp_prepare_data(in_dir, feedback_name)

        # 去除nan, 预处理, 筛选过少轨迹的用户
        def tmp(normal):
            normal = normal.dropna()
            normal['page_name'] = normal['page_name'].map(url_prepreprocess)
            normal = normal.dropna()
            return normal

        normal = tmp(normal)
        feedback = tmp(feedback)


if __name__ == "__main__":

    in_dir = pre_args.in_dir_prepre
    normal_names = pre_args.normal_names_prepre
    feedback_names = pre_args.feedback_names_prepre
    output_dir = pre_args.output_dir_prepre
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    g_lowercase2uppercase(in_dir, normal_names, feedback_names, output_dir)
    # 保存lowercase2uppercase
    with open(os.path.join(output_dir, "lowercase2uppercase.json"), "w") as f:
        json.dump(lowercase2uppercase, f, indent=4)