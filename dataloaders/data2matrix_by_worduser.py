"""数据加载器
"""
import os
import itertools
import pandas as pd
import torch
import json
import collections
import re

def tmp_prepare_data(in_dir, file_name):
    normal_data_path = os.path.join(in_dir, file_name)
    df = pd.read_csv(normal_data_path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.reset_index()
    df.rename(columns={"index": "unique_id"}, inplace=True)
    return df

def prepare_normal_data(args, **kwargs):
    """载入正常数据
    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数

    Returns:
        feature_list (torch.LongTensor): 异常用户的session向量
        feature_len_list (torch.LongTensor): 异常用户的session向量长度
        feature_sid_list (pd.DataFrame): 用户的session_id
        feature_uid_list (pd.DataFrame): 用户的user_id
        feature_label_list (torch.LongTensor): 异常用户的session向量标签
        unknown_page_name (collections.Counter): 未知页面名
        unknown_page_len (collections.Counter): 超过max_seq_len的页面长度
    """
    word2idx = json.load(open(args.vocab_dict_path, 'r'))
    max_seq_len = args.max_seq_len

    df_normal= tmp_prepare_data(args.dataset_root, args.file_name_normal)
    print('正常用户：根据session中的页面筛选前用户的轨迹数为：', len(df_normal))

    # 建立每一个session的id序列，比如两个session，总的页面数是3，那么生成一个二维列表[[1,2,0], [3,4,len(all_page_name)]]
    unknown_page_name = []
    unknown_page_len = []
    feature_list = []
    feature_len_list = []
    feature_sid_list = []  # session_id
    feature_uid_list = []  # user_id
    df_normal.sort_values(['date_time'], ascending=[True], inplace=True)
    for n, en in df_normal.groupby("user_id"):
        feature_sid_list.append(n)
        cur_feature = []
        first = True
        cur_uid = None
        for _, e in en.iterrows():
            if first:
                cur_uid = e["user_id"]
                first = False
            words = [word.lower() for word in re.split(r'[^a-zA-Z0-9]', e["page_name"]) if word]
            for word in words:
                if word in word2idx:
                    cur_feature.append(word2idx[word])
                else:
                    cur_feature.append(word2idx['<unk>'])
                    unknown_page_name.append(word)
        feature_uid_list.append(cur_uid)
        if len(cur_feature) >= max_seq_len:
            unknown_page_len.append(len(cur_feature))
            cur_feature = cur_feature[-(max_seq_len-1):]
        cur_feature.append(word2idx['<eos>'])
        len_cur_feature = len(cur_feature)
        cur_feature.extend(
            [word2idx['<pad>']] * (max_seq_len - len_cur_feature))
        # 可以在此加入其他特征
        feature_list.append(cur_feature)
        feature_len_list.append(len_cur_feature)

    # 转化成[batch_size, seq_len]
    feature_list = list(itertools.zip_longest(
        *feature_list, fillvalue=word2idx['<pad>']))
    feature_list = torch.LongTensor(
        feature_list).T  # shape: [batch_size, seq_len]
    feature_len_list = torch.LongTensor(
        feature_len_list)  # shape: [batch_size]
    feature_sid_list = pd.DataFrame(
        feature_sid_list, columns=['session_id'])  # shape: [batch_size]
    feature_uid_list = pd.DataFrame(
        feature_uid_list, columns=['user_id'])  # shape: [batch_size]
    feature_label_list = torch.zeros(  # 标签全部为0
        feature_list.shape[0], dtype=feature_list.dtype)  # shape: [batch_size]

    unknown_page_name = collections.Counter(unknown_page_name)
    unknown_page_len = collections.Counter(unknown_page_len)
    return feature_list, feature_len_list, feature_sid_list, feature_uid_list, feature_label_list, unknown_page_name, unknown_page_len


def prepare_abnormal_data(args, **kwargs):
    """载入异常数据
    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数

    Returns:
        feature_list (torch.LongTensor): 异常用户的session向量
        feature_len_list (torch.LongTensor): 异常用户的session向量长度
        feature_sid_list (pd.DataFrame): 用户的session_id
        feature_uid_list (pd.DataFrame): 用户的user_id
        feature_label_list (torch.LongTensor): 异常用户的session向量标签
        unknown_page_name (collections.Counter): 未知页面名
        unknown_page_len (collections.Counter): 超过max_seq_len的页面长度
    """
    word2idx = json.load(open(args.vocab_dict_path, 'r'))
    max_seq_len = args.max_seq_len

    df_abnormal= tmp_prepare_data(args.dataset_root, args.file_name_abnormal)
    print('异常用户：根据session中的页面筛选前用户的轨迹数为：', len(df_abnormal))

    # 建立每一个session的id序列，比如两个session，总的页面数是3，那么生成一个二维列表[[1,2,0], [3,4,len(all_page_name)]]
    unknown_page_name = []
    unknown_page_len = []
    feature_list = []
    feature_len_list = []
    feature_sid_list = []
    feature_uid_list = []
    df_abnormal.sort_values(['date_time'], ascending=[True], inplace=True)
    for n, en in df_abnormal.groupby("session_id"):
        feature_sid_list.append(n)
        cur_feature = []
        first = True
        cur_uid = None
        for _, e in en.iterrows():
            if first:
                cur_uid = e["user_id"]
                first = False
            words = [word.lower() for word in re.split(r'[^a-zA-Z0-9]', e["page_name"]) if word]
            for word in words:
                if word in word2idx:
                    cur_feature.append(word2idx[word])
                else:
                    cur_feature.append(word2idx['<unk>'])
                    unknown_page_name.append(word)
        feature_uid_list.append(cur_uid)
        if len(cur_feature) >= max_seq_len:
            unknown_page_len.append(len(cur_feature))
            cur_feature = cur_feature[-(max_seq_len-1):]
        cur_feature.append(word2idx['<eos>'])
        len_cur_feature = len(cur_feature)
        cur_feature.extend(
            [word2idx['<pad>']] * (max_seq_len - len_cur_feature))
        # 可以在此加入其他特征
        feature_list.append(cur_feature)
        feature_len_list.append(len_cur_feature)

    # 转化成[batch_size, seq_len]
    feature_list = list(itertools.zip_longest(
        *feature_list, fillvalue=word2idx['<pad>']))
    feature_list = torch.LongTensor(
        feature_list).T  # shape: [batch_size, seq_len]
    feature_len_list = torch.LongTensor(
        feature_len_list)  # shape: [batch_size]
    feature_sid_list = pd.DataFrame(
        feature_sid_list, columns=['session_id'])  # shape: [batch_size]
    feature_uid_list = pd.DataFrame(
        feature_uid_list, columns=['user_id'])  # shape: [batch_size]
    feature_label_list = torch.ones(
        feature_list.shape[0], dtype=feature_list.dtype)  # shape: [batch_size]

    unknown_page_name = collections.Counter(unknown_page_name)
    unknown_page_len = collections.Counter(unknown_page_len)
    return feature_list, feature_len_list, feature_sid_list, feature_uid_list, feature_label_list, unknown_page_name, unknown_page_len

if __name__ == '__main__':
    pass
