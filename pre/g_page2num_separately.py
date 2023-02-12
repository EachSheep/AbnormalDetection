"""首先将原始数据处理成我们期望的表的形式保存下来
"""
import os
import pandas as pd
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
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.reset_index()
    df.rename(columns={"index": "unique_id"}, inplace=True)
    return df

def preprocess(in_dir, normal_names, feedback_names, output_dir, **kwargs):
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
    page2num_names = kwargs["page2num_names"]

    for normal_name, feedback_name, page2num_name in zip(normal_names, feedback_names, page2num_names):

        normal = tmp_prepare_data(in_dir, normal_name)
        print('正常用户：根据session中的页面筛选前用户的轨迹数为：', len(normal))
        # 生成page2num文件
        normal = normal.groupby('page_name')[
            'unique_id'].count().sort_values(ascending=False)
        normal_p2n = dict(
            zip(normal.index, [int(value) for value in normal.values]))
        json.dump(normal_p2n, open(
            os.path.join(output_dir, '.'.join(page2num_name.split('.')[:-1]) + "_normal.json"), "w"), indent=4)

        feedback = tmp_prepare_data(in_dir, feedback_name)
        print('异常用户：根据session中的页面筛选前用户的轨迹数为：', len(feedback))
        feedback = feedback.groupby('page_name')[
            'unique_id'].count().sort_values(ascending=False)
        feedback_p2n = dict(
            zip(feedback.index, [int(value) for value in feedback.values]))
        json.dump(feedback_p2n, open(
            os.path.join(output_dir, '.'.join(page2num_name.split('.')[:-1]) + "_feedback.json"), "w"), indent=4)

if __name__ == "__main__":

    in_dir = pre_args.in_dir_gen
    normal_names = pre_args.normal_names_gen
    feedback_names = pre_args.feedback_names_gen
    output_dir = pre_args.output_dir_gen
    kwargs = {
        "page2num_names": pre_args.page2num_names_gen
    }
    preprocess(in_dir, normal_names, feedback_names, output_dir, **kwargs)