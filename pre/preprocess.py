"""首先将原始数据处理成我们期望的表的形式保存下来
"""
import os
import pandas as pd

from pre_argparser import pre_args
from UrlPreprocess import url_preprocess

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
    min_seq_len = kwargs["min_seq_len"]

    for normal_name, feedback_name in zip(normal_names, feedback_names):

        normal = tmp_prepare_data(in_dir, normal_name)
        print('正常用户：根据session中的页面筛选前用户的轨迹数为：', len(normal))
        feedback = tmp_prepare_data(in_dir, feedback_name)
        print('异常用户：根据session中的页面筛选前用户的轨迹数为：', len(feedback))

        # 去除nan, 预处理, 筛选过少轨迹的用户
        def tmp(normal):
            normal['page_name'] = normal['page_name'].map(url_preprocess)
            
            df_normal_cnt = normal.groupby("session_id")['unique_id'].count()
            df_normal_sel_id = df_normal_cnt[df_normal_cnt >= min_seq_len]
            normal = normal[normal['session_id'].isin(
                df_normal_sel_id.index)]
            return normal

        normal = tmp(normal)
        feedback = tmp(feedback)

        normal = normal.drop(columns=["unique_id"])
        feedback = feedback.drop(columns=["unique_id"])
        normal.to_csv(os.path.join(output_dir, normal_name), index=False)
        feedback.to_csv(os.path.join(output_dir, feedback_name), index=False)


if __name__ == "__main__":

    in_dir = pre_args.in_dir_pre
    normal_names = pre_args.normal_names_pre
    feedback_names = pre_args.feedback_names_pre
    output_dir = pre_args.output_dir_pre
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    kwargs = {
        "min_seq_len": 10,
    }
    preprocess(in_dir, normal_names, feedback_names, output_dir, **kwargs)
