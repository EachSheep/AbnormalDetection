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
    # df["date_time"] = pd.to_datetime(df["date_time"])
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
    tmp_url = re.split(r'[^a-zA-Z]', url)
    tmp_url = [cur for cur in tmp_url if cur]
    global lowercase2uppercase
    [lowercase2uppercase.setdefault(cur.lower(), cur) for cur in tmp_url if cur != cur.lower()]

    url = url.lower()
    # 去除url中的汉字, 去除所有的url中文编码, 去除所有逗号, 去除所有~
    url = re.sub(r'[\u4e00-\u9fa5]|%[a-fA-F\d]{2}|~|,', '', url)
    # "https://m.amap.com/navigation/carmap/&saddr=121.834638%2c29.847424%2c%e6%88%91%e7%9a%84%e4%bd%8d%e7%bd%ae&daddr=121.51234007%2c29.84995423%2c%e5%ae%81%e6%b3%a2%e5%ae%a2%e8%bf%90%e4%b8%ad%e5%bf%83%e5%9c%b0%e9%93%81%e7%ab%99c%e5%8f%a3"
    # 变成：https://m.amap.com/navigation/carmap/&saddr=daddr=
    url = re.sub(r'=.*&|=.*$', '=', url)
    url = re.sub(r'/+$|=+$|-+$|\_+$|\?+$', '', url)  # 末尾不能以/, =, -, _, ?结尾
    return url


def prepreprocess(in_dir, normal_names, feedback_names, output_dir, **kwargs):
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

        # 去除nan, 预处理, 筛选过少轨迹的用户
        def tmp(normal):
            normal = normal.dropna()
            normal['page_name'] = normal['page_name'].map(url_prepreprocess)
            normal = normal.dropna()
            return normal

        normal = tmp(normal)
        feedback = tmp(feedback)

        normal = normal.drop(columns=["unique_id"])
        feedback = feedback.drop(columns=["unique_id"])
        normal.to_csv(os.path.join(output_dir, normal_name), index=False)
        feedback.to_csv(os.path.join(output_dir, feedback_name), index=False)


def prepreprocess_from_page2num(json_data):
    """
    """
    
    keys = list(json_data.keys())
    values = list(json_data.values())
    keys = list(map(url_prepreprocess, keys))
    json_data = dict(zip(keys, values))

    def merge(page2num):
        tmp = {}
        for key, value in page2num.items():
            tmp[key] = tmp.get(key, 0) + value
        return tmp
    json_data = merge(json_data)
    return json_data


if __name__ == "__main__":

    in_dir = pre_args.in_dir_prepre
    normal_names = pre_args.normal_names_prepre
    feedback_names = pre_args.feedback_names_prepre
    output_dir = pre_args.output_dir_prepre
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    prepreprocess(in_dir, normal_names, feedback_names, output_dir)
    
    # 保存lowercase2uppercase
    with open(os.path.join(output_dir, "lowercase2uppercase.json"), "w") as f:
        json.dump(lowercase2uppercase, f, indent=4)

    # file_path = "../experiment/page2nums/page2num-1.json"
    # json_data = json.load(open(file_path, "r"))
    # json_data = prepreprocess_from_page2num(json_data)
    # json.dump(json_data, open("../experiment/page2nums/page2num.json", "w"), indent=4)
