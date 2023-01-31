"""数据加载器
"""
import os
import itertools
import pandas as pd
import torch
import json
import collections
from torch.utils.data import DataLoader

from dataloaders.MyDataset import MyDataset
from dataloaders.utlis import worker_init_fn_seed, BalancedBatchSampler, RandomedBatchSampler
from pre.wash_pagename import *


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
    page2id = json.load(open(args.vocab_dict_path, 'r'))
    max_seq_len = args.max_seq_len + 1  # +1是<eos>的位置

    normal_data_path = os.path.join(args.dataset_root, args.file_name_normal) # 载入数据
    df_normal = pd.read_csv(normal_data_path)
    df_normal["date_time"] = pd.to_datetime(df_normal.date_time)
    df_normal = df_normal.reset_index()
    df_normal.rename(columns={"index": "unique_id"}, inplace=True)
    print('正常用户：根据session中的页面筛选前用户的轨迹数为：', len(df_normal))


    # 首先去掉和feedback数据交叉的部分，这部分是，特殊代码，不通用
    abnormal_data_path = os.path.join(
        args.dataset_root, args.file_name_abnormal)
    df_abnormal = pd.read_csv(abnormal_data_path)
    df_abnormal["date_time"] = pd.to_datetime(df_abnormal.date_time)
    df_abnormal = df_abnormal.reset_index()
    df_abnormal.rename(columns={"index": "unique_id"}, inplace=True)
    print('异常用户：根据session中的页面筛选前用户的轨迹数为：', len(df_abnormal))
    # 去除normal中session_id的交集
    df_normal = df_normal[~df_normal["session_id"].isin(df_abnormal["session_id"].unique())]

    # 去除normal中的nan
    df_normal = df_normal.dropna(subset=['page_name'])

    # 然后对数据中的page_name进行清洗
    df_normal['page_name'] = df_normal['page_name'].map(preprocess)
    # 先给不是url、在extension_of_filename中，在lastword_dict、freq_lastword_dict中的的数据打上标签
    df_normal['is_ok'] = df_normal['page_name'].map(filter_by_url)
    # 给在freq_url_dict中的数据打上标签
    df_normal['is_ok'] = df_normal['is_ok'] + df_normal['page_name'].map(filter_by_freq_url)
    # 给is_ok == False的page_name做一次process_by_force的操作
    df_nomral_no_process = df_normal[df_normal['is_ok'] == True]
    df_normal_need_process = df_normal[df_normal['is_ok'] == False]
    df_normal_need_process.loc[:, ['page_name']] = df_normal_need_process['page_name'].map(process_by_force)
    df_normal = pd.concat([df_nomral_no_process, df_normal_need_process])
    
    # 根据sessuion中页面数的CDF图，根据这个图决定筛掉用户轨迹小于多少的用户数据。
    df_normal_cnt = df_normal.groupby("session_id")['unique_id'].count()
    df_normal_larger_id = df_normal_cnt[df_normal_cnt >= args.filter_num]
    df_normal = df_normal[df_normal['session_id'].isin(
        df_normal_larger_id.index)]

    # 建立每一个session的id序列，比如两个session，总的页面数是3，那么生成一个二维列表[[1,2,0], [3,4,len(all_page_name)]]
    unknown_page_name = []
    unknown_page_len = []
    feature_list = []
    feature_len_list = []
    feature_sid_list = []  # session_id
    feature_uid_list = []  # user_id
    df_normal.sort_values(['date_time'], ascending=[True], inplace=True)
    # df_normal.sort_values(['date_time', 'sort_source'], ascending=[True, True], inplace=True)
    for n, en in df_normal.groupby("session_id"):
        feature_sid_list.append(n)
        cur_feature = []
        first = True
        cur_uid = None
        for _, e in en.iterrows():
            if first:
                cur_uid = e.user_id
                first = False
            if e.page_name in page2id:
                cur_feature.append(page2id[e.page_name])
            else:
                cur_feature.append(page2id['<unk>'])
                unknown_page_name.append(e.page_name)
        feature_uid_list.append(cur_uid)
        if len(cur_feature) >= max_seq_len:
            unknown_page_len.append(len(cur_feature))
            cur_feature = cur_feature[-(max_seq_len-1):]
        cur_feature.append(page2id['<eos>'])
        len_cur_feature = len(cur_feature)
        cur_feature.extend(
            [page2id['<pad>']] * (max_seq_len - len_cur_feature))
        # 可以在此加入其他特征
        feature_list.append(cur_feature)
        feature_len_list.append(len_cur_feature)

    # 转化成[batch_size, seq_len]
    feature_list = list(itertools.zip_longest(
        *feature_list, fillvalue=page2id['<pad>']))
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
    page2id = json.load(open(args.vocab_dict_path, 'r'))
    max_seq_len = args.max_seq_len + 1

    abnormal_data_path = os.path.join(
        args.dataset_root, args.file_name_abnormal)
    df_abnormal = pd.read_csv(abnormal_data_path)
    df_abnormal["date_time"] = pd.to_datetime(df_abnormal.date_time)
    df_abnormal = df_abnormal.reset_index()
    df_abnormal.rename(columns={"index": "unique_id"}, inplace=True)
    print('异常用户：根据session中的页面筛选前用户的轨迹数为：', len(df_abnormal))

    # 去除normal中的nan
    df_abnormal = df_abnormal.dropna(subset=['page_name'])

    # 然后对数据中的page_name进行清洗
    df_abnormal['page_name'] = df_abnormal['page_name'].map(preprocess)
    # 先给不是url、在extension_of_filename中，在lastword_dict、freq_lastword_dict中的的数据打上标签
    df_abnormal['is_ok'] = df_abnormal['page_name'].map(filter_by_url)
    # 给在freq_url_dict中的数据打上标签
    df_abnormal['is_ok'] = df_abnormal['is_ok'] + df_abnormal['page_name'].map(filter_by_freq_url)
    # 给is_ok == False的page_name做一次process_by_force的操作
    df_abnomral_no_process = df_abnormal[df_abnormal['is_ok'] == True]
    df_abnormal_need_process = df_abnormal[df_abnormal['is_ok'] == False]
    df_abnormal_need_process.loc[:, ['page_name']] = df_abnormal_need_process['page_name'].map(process_by_force)
    df_abnormal = pd.concat([df_abnomral_no_process, df_abnormal_need_process])

    # 根据sessuion中页面数的CDF图，根据这个图决定筛掉用户轨迹小于多少的用户数据。
    df_abnormal_cnt = df_abnormal.groupby("session_id")['unique_id'].count()
    df_abnormal_larger_id = df_abnormal_cnt[df_abnormal_cnt >= args.filter_num]
    df_abnormal = df_abnormal[df_abnormal['session_id'].isin(
        df_abnormal_larger_id.index)]
    print('异常用户：根据session中的页面筛选后用户的轨迹数为：', len(df_abnormal))

    # 建立每一个session的id序列，比如两个session，总的页面数是3，那么生成一个二维列表[[1,2,0], [3,4,len(all_page_name)]]
    unknown_page_name = []
    unknown_page_len = []
    feature_list = []
    feature_len_list = []
    feature_sid_list = []
    feature_uid_list = []
    df_abnormal.sort_values(['date_time'], ascending=[True], inplace=True)
    # df_abnormal.sort_values(['date_time', 'sort_source'], ascending=[True, True], inplace=True)
    for n, en in df_abnormal.groupby("session_id"):
        feature_sid_list.append(n)
        cur_feature = []
        first = True
        cur_uid = None
        for _, e in en.iterrows():
            if first:
                cur_uid = e.user_id
                first = False
            if e.page_name in page2id:
                cur_feature.append(page2id[e.page_name])
            else:
                cur_feature.append(page2id['<unk>'])
                unknown_page_name.append(e.page_name)
        feature_uid_list.append(cur_uid)
        if len(cur_feature) >= max_seq_len:
            unknown_page_len.append(len(cur_feature))
            cur_feature = cur_feature[-(max_seq_len-1):]
        cur_feature.append(page2id['<eos>'])
        len_cur_feature = len(cur_feature)
        cur_feature.extend(
            [page2id['<pad>']] * (max_seq_len - len_cur_feature))
        # 可以在此加入其他特征
        feature_list.append(cur_feature)
        feature_len_list.append(len_cur_feature)

    # 转化成[batch_size, seq_len]
    feature_list = list(itertools.zip_longest(
        *feature_list, fillvalue=page2id['<pad>']))
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


def prepare_train_data(args, **kwargs):
    """载入数据，划分训练集和测试集，划分的依据是args.train_ratio

    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数
    Returns:
        train_data (torch.LongTensor): 训练集的session向量
        train_len (torch.LongTensor): 训练集的session向量长度
        train_sid (pd.DataFrame): 训练集的session_id
        train_uid (pd.DataFrame): 训练集的user_id
        train_label (torch.LongTensor): 训练集的session向量标签
        test_data (torch.LongTensor): 测试集的session向量
        test_len (torch.LongTensor): 测试集的session向量长度
        test_id (pd.DataFrame): 测试集的session_id
        test_uid (pd.DataFrame): 测试集的user_id
        test_label (torch.LongTensor): 测试集的session向量标签
    """
    feature_normal, feature_len_normal, feature_sid_normal, feature_uid_normal, feature_label_normal, unknown_page_name_normal, unknown_page_len_normal = prepare_normal_data(
        args, **kwargs)
    feature_abnormal, feature_len_abnormal, feature_sid_abnormal, feature_uid_abnormal, feature_label_abnormal, unknown_page_name_abnormal, unknown_page_len_abnormal = prepare_abnormal_data(
        args, **kwargs)
    unknown_page_name_normal.update(unknown_page_name_abnormal)
    unknown_page_len_normal.update(unknown_page_len_abnormal)
    unknown_page_name = unknown_page_name_normal
    unknown_page_len = unknown_page_len_normal
    del unknown_page_name_normal, unknown_page_len_normal, unknown_page_name_abnormal, unknown_page_len_abnormal

    if not os.path.exists(os.path.join(args.experiment_dir, 'jsons')):
        os.makedirs(os.path.join(args.experiment_dir, 'jsons'))
    json.dump(unknown_page_name, open(os.path.join(args.experiment_dir,
              'jsons', f'unknown_page_name-train-{args.cur_time}.json'), 'w'), indent=4)
    json.dump(unknown_page_len, open(os.path.join(args.experiment_dir,
              'jsons', f'unknown_page_len-train-{args.cur_time}.json'), 'w'), indent=4)

    # 按照真实数据集的比例混合正常和异常数据，真实数据集的比例即 异常用户数 / 正常用户数
    # 这里假定session_vector_list和session_vector_list_abnormal就是真实数据集，因此不需要进行抽样，直接按照比例混合即可
    torch.manual_seed(args.random_seed)

    # 抽样正常数据
    normal_index = torch.randperm(feature_normal.shape[0])
    train_normal_index = normal_index[:int(
        feature_normal.shape[0] * args.train_ratio)]
    test_normal_index = normal_index[int(
        feature_normal.shape[0] * args.train_ratio):]
    train_data_normal = feature_normal[train_normal_index]
    train_len_normal = feature_len_normal[train_normal_index]
    train_sid_normal = feature_sid_normal.iloc[train_normal_index.tolist(), :]
    train_uid_normal = feature_uid_normal.iloc[train_normal_index.tolist(
    ), :]
    train_label_normal = feature_label_normal[train_normal_index]
    test_normal_data = feature_normal[test_normal_index]
    test_normal_len = feature_len_normal[test_normal_index]
    test_normal_id = feature_sid_normal.iloc[test_normal_index.tolist(), :]
    test_normal_user_id = feature_uid_normal.iloc[test_normal_index.tolist(
    ), :]
    test_normal_label = feature_label_normal[test_normal_index]

    # 抽样异常数据
    abnormal_index = torch.randperm(feature_abnormal.shape[0])
    train_abnormal_index = abnormal_index[:int(
        feature_abnormal.shape[0] * args.train_ratio)]
    test_abnormal_index = abnormal_index[int(
        feature_abnormal.shape[0] * args.train_ratio):]
    train_data_abnormal = feature_abnormal[train_abnormal_index]
    train_len_abnormal = feature_len_abnormal[train_abnormal_index]
    train_sid_abnormal = feature_sid_abnormal.iloc[train_abnormal_index.tolist(
    ), :]
    train_uid_abnormal = feature_uid_abnormal.iloc[train_abnormal_index.tolist(
    ), :]
    train_label_abnormal = feature_label_abnormal[train_abnormal_index]
    test_abnormal_data = feature_abnormal[test_abnormal_index]
    test_abnormal_len = feature_len_abnormal[test_abnormal_index]
    test_abnormal_id = feature_sid_abnormal.iloc[test_abnormal_index.tolist(
    ), :]
    test_abnormal_user_id = feature_uid_abnormal.iloc[test_abnormal_index.tolist(
    ), :]
    test_abnormal_label = feature_label_abnormal[test_abnormal_index]

    # 合并训练数据
    train_data = torch.cat((train_data_normal, train_data_abnormal), dim=0)
    train_len = torch.cat((train_len_normal, train_len_abnormal), dim=0)
    train_sid = pd.concat((train_sid_normal, train_sid_abnormal), axis=0)
    train_uid = pd.concat(
        (train_uid_normal, train_uid_abnormal), axis=0)
    train_label = torch.cat((train_label_normal, train_label_abnormal), dim=0)
    train_sid = train_sid.reset_index(drop=True)
    train_uid = train_uid.reset_index(drop=True)

    # 合并测试数据
    test_data = torch.cat((test_normal_data, test_abnormal_data), dim=0)
    test_len = torch.cat((test_normal_len, test_abnormal_len), dim=0)
    test_id = pd.concat((test_normal_id, test_abnormal_id), axis=0)
    test_uid = pd.concat(
        (test_normal_user_id, test_abnormal_user_id), axis=0)
    test_label = torch.cat((test_normal_label, test_abnormal_label), dim=0)
    test_id = test_id.reset_index(drop=True)
    test_uid = test_uid.reset_index(drop=True)

    return train_data, train_len, train_sid, train_uid, train_label, test_data, test_len, test_id, test_uid, test_label


def build_train_dataloader(args, **kwargs):
    """构建dataloader

    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数
    Returns:
        train_loader (torch.utils.data.DataLoader): 训练集的dataloader
        test_loader (torch.utils.data.DataLoader): 测试集的dataloader
    """
    train_data, train_len, train_sid, train_uid, train_label, test_data, test_len, test_id, test_uid, test_label = prepare_train_data(
        args, **kwargs)
    train_set = MyDataset(args, train_data, train_label,
                          **{"len": train_len, "sid": train_sid, "uid": train_uid})
    test_set = MyDataset(args, test_data, test_label, **
                         {"len": test_len, "sid": test_id, "uid": test_uid})
    train_loader = DataLoader(
        train_set,
        worker_init_fn=worker_init_fn_seed,
        batch_sampler=BalancedBatchSampler(args, train_set),
        **kwargs
    )
    test_loader = DataLoader(
        test_set,
        # batch_size=args.batch_size,
        # shuffle=False,
        worker_init_fn=worker_init_fn_seed,
        batch_sampler=RandomedBatchSampler(args, test_set),
        **kwargs
    )
    return train_loader, test_loader


def prepare_test_data(args, **kwargs):
    """载入数据，划分训练集和测试集，划分的依据是args.train_ratio

    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数
    Returns:
        test_data (torch.LongTensor): 测试集的session向量
        test_len (torch.LongTensor): 测试集的session向量长度
        test_id (pd.DataFrame): 测试集的session_id
        test_uid (pd.DataFrame): 测试集的user_id
        test_label (torch.LongTensor): 测试集的session向量标签
    """
    data_normal, len_normal, sid_normal, uid_normal, label_normal, unknown_page_name_normal, unknown_page_len_normal = prepare_normal_data(
        args, **kwargs)
    data_abnormal, len_abnormal, sid_abnormal, uid_abnormal, label_abnormal, unknown_page_name_abnormal, unknown_page_len_abnormal = prepare_abnormal_data(
        args, **kwargs)
    unknown_page_name_normal.update(unknown_page_name_abnormal)
    unknown_page_len_normal.update(unknown_page_len_abnormal)
    unknown_page_name = unknown_page_name_normal
    unknown_page_len = unknown_page_len_normal
    del unknown_page_name_normal, unknown_page_len_normal, unknown_page_name_abnormal, unknown_page_len_abnormal

    if not os.path.exists(os.path.join(args.experiment_dir, 'jsons')):
        os.makedirs(os.path.join(args.experiment_dir, 'jsons'))
    json.dump(unknown_page_name, open(os.path.join(args.experiment_dir,
              'jsons', f'unknown_page_name-test-{args.cur_time}.json'), 'w'), indent=4)
    json.dump(unknown_page_len, open(os.path.join(args.experiment_dir,
              'jsons', f'unknown_page_len-test-{args.cur_time}.json'), 'w'), indent=4)

    # 合并训练数据
    data = torch.cat((data_normal, data_abnormal), dim=0)
    lenn = torch.cat((len_normal, len_abnormal), dim=0)
    sid = pd.concat((sid_normal, sid_abnormal), axis=0)
    uid = pd.concat((uid_normal, uid_abnormal), axis=0)
    label = torch.cat((label_normal, label_abnormal), dim=0)
    sid = sid.reset_index(drop=True)
    uid = uid.reset_index(drop=True)

    return data, lenn, sid, uid, label

def build_test_dataloader(args, **kwargs):
    """构建dataloader

    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数
    Returns:
        test_loader (torch.utils.data.DataLoader): 测试集的dataloader
    """
    test_data, test_len, test_id, test_uid, test_label = prepare_test_data(
        args, **kwargs)
    test_set = MyDataset(args, test_data, test_label, **
                         {"len": test_len, "sid": test_id, "uid": test_uid})
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn_seed,
        **kwargs
    )
    return test_loader

if __name__ == '__main__':
    pass
