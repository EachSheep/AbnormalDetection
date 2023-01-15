"""数据加载器
"""
import os
import itertools
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataloaders.MyDataset import MyDataset
from dataloaders.utlis import worker_init_fn_seed, BalancedBatchSampler


def prepare_normal_data(args, **kwargs):
    """载入正常数据
    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数

    Returns:
        session_vector_list (torch.LongTensor): 异常用户的session向量
        session_vector_len_list (torch.LongTensor): 异常用户的session向量长度
        session_vector_label_list (torch.LongTensor): 异常用户的session向量标签
        all_page_name2id (dict): 所有页面的名称到id的映射
        max_df_normal_cnt (int): 正常用户的session中页面的最大值
    """
    # normal_data_path = os.path.join(
    #     kwargs['dataset_root'], kwargs['file_name_normal'])
    normal_data_path = os.path.join(args.dataset_root, args.file_name_normal)
    df_normal = pd.read_csv(normal_data_path)
    df_normal["date_time"] = pd.to_datetime(df_normal.date_time)
    df_normal = df_normal.reset_index()
    df_normal.rename(columns={"index": "unique_id"}, inplace=True)
    print('正常用户：根据session中的页面筛选前用户的轨迹数为：', len(df_normal))

    # 给所有的用户页面编码
    all_page_name = df_normal['page_name'].unique()
    all_page_name2id = dict(zip(all_page_name, range(len(all_page_name))))
    all_page_name2id['<eos>'] = len(all_page_name)
    all_page_name2id['<pad>'] = len(all_page_name) + 1
    all_page_name2id['<unk>'] = len(all_page_name) + 2  # 未知页面

    # 根据sessuion中页面数的CDF图，根据这个图决定筛掉用户轨迹小于多少的用户数据。暂定10
    df_normal_cnt = df_normal.groupby("session_id")['unique_id'].count()
    df_normal_larger_id = df_normal_cnt[df_normal_cnt >= args.filter_num]
    df_normal = df_normal[df_normal['session_id'].isin(
        df_normal_larger_id.index)]
    max_df_normal_cnt = df_normal_cnt.max() + 1

    # 建立每一个session的id序列，比如两个session，总的页面数是3，那么生成一个二维列表[[1,2,0], [3,4,len(all_page_name)]]
    session_vector_list = []
    session_vector_len_list = []
    df_normal.sort_values(['date_time'], ascending=[True], inplace=True)
    # df_normal.sort_values(['date_time', 'sort_source'], ascending=[True, True], inplace=True)
    for n, en in df_normal.groupby("session_id"):
        cur_session_vector = []
        for _, e in en.iterrows():
            cur_session_vector.append(all_page_name2id[e.page_name])
        cur_session_vector.append(all_page_name2id['<eos>'])
        len_cur_session_vector = len(cur_session_vector)
        cur_session_vector.extend(
            [all_page_name2id['<pad>']] * (max_df_normal_cnt - len_cur_session_vector))
        session_vector_list.append(cur_session_vector)
        session_vector_len_list.append(len_cur_session_vector)

    # 转化成[seq_len,batch_size]
    session_vector_list = list(itertools.zip_longest(
        session_vector_list, fillvalue=all_page_name2id['<pad>']))
    session_vector_list = torch.LongTensor(
        session_vector_list)  # shape: [seq_len, batch_size]
    session_vector_len_list = torch.LongTensor(
        session_vector_len_list)  # shape: [batch_size]
    session_vector_label_list = torch.zeros( # 标签全部为0
        session_vector_list.shape[1], dtype=session_vector_list.dtype)  # shape: [batch_size]

    return session_vector_list, session_vector_len_list, session_vector_label_list, all_page_name2id, max_df_normal_cnt


def prepare_abnormal_data(args, **kwargs):
    """载入异常数据
    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数

    Returns:
        session_vector_list (torch.LongTensor): 异常用户的session向量
        session_vector_len_list (torch.LongTensor): 异常用户的session向量长度
        session_vector_label_list (torch.LongTensor): 异常用户的session向量标签
        all_page_name2id (dict): 所有页面的名称到id的映射
        max_df_normal_cnt (int): 正常用户的session中页面的最大值
    """
    all_page_name2id = kwargs['all_page_name2id']
    max_df_normal_cnt = kwargs['max_df_normal_cnt']

    # abnormal_data_path = os.path.join(
    #     kwargs['dataset_root'], kwargs['file_name_abnormal'])
    abnormal_data_path = os.path.join(
        args.dataset_root, args.file_name_abnormal)
    df_abnormal = pd.read_csv(abnormal_data_path)
    df_abnormal["date_time"] = pd.to_datetime(df_abnormal.date_time)
    df_abnormal = df_abnormal.reset_index()
    df_abnormal.rename(columns={"index": "unique_id"}, inplace=True)
    print('异常用户：根据session中的页面筛选前用户的轨迹数为：', len(df_abnormal))

    # 根据sessuion中页面数的CDF图，根据这个图决定筛掉用户轨迹小于多少的用户数据。暂定10
    df_abnormal_cnt = df_abnormal.groupby("session_id")['unique_id'].count()
    df_abnormal_larger_id = df_abnormal_cnt[df_abnormal_cnt >= args.filter_num]
    df_abnormal = df_abnormal[df_abnormal['session_id'].isin(
        df_abnormal_larger_id.index)]
    print('异常用户：根据session中的页面筛选后用户的轨迹数为：', len(df_abnormal))

    # 建立每一个session的id序列，比如两个session，总的页面数是3，那么生成一个二维列表[[1,2,0], [3,4,len(all_page_name)]]
    session_vector_list = []
    session_vector_len_list = []
    df_abnormal.sort_values(['date_time'], ascending=[True], inplace=True)
    # df_abnormal.sort_values(['date_time', 'sort_source'], ascending=[True, True], inplace=True)
    for n, en in df_abnormal.groupby("session_id"):
        cur_session_vector = []
        for _, e in en.iterrows():
            cur_session_vector.append(all_page_name2id[e.page_name])
        cur_session_vector.append(all_page_name2id['<eos>'])
        len_cur_session_vector = len(cur_session_vector)
        cur_session_vector.extend(
            [all_page_name2id['<pad>']] * (max_df_normal_cnt - len_cur_session_vector))
        session_vector_list.append(cur_session_vector)
        session_vector_len_list.append(len_cur_session_vector)

    # 转化成[seq_len,batch_size]
    session_vector_list = list(itertools.zip_longest(
        session_vector_list, fillvalue=all_page_name2id['<pad>']))
    session_vector_list = torch.LongTensor(
        session_vector_list)  # shape: [seq_len, batch_size]
    session_vector_len_list = torch.LongTensor(
        session_vector_len_list)  # shape: [batch_size]
    session_vector_label_list = torch.ones(
        session_vector_list.shape[1], dtype=session_vector_list.dtype)  # shape: [batch_size]

    return session_vector_list, session_vector_len_list, session_vector_label_list, all_page_name2id, max_df_normal_cnt


def prepare_data(args, **kwargs):
    """载入数据，划分训练集和测试集，划分的依据是args.train_ratio

    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数
    Returns:
        train_data (torch.LongTensor): 训练集的session向量
        train_len (torch.LongTensor): 训练集的session向量长度
        train_label (torch.LongTensor): 训练集的session向量标签
        test_data (torch.LongTensor): 测试集的session向量
        test_len (torch.LongTensor): 测试集的session向量长度
        test_label (torch.LongTensor): 测试集的session向量标签
    """
    session_vector_list_normal, session_vector_len_list_normal, session_vector_label_list_normal, all_page_name2id, max_df_normal_cnt = prepare_normal_data(
        args, **kwargs)
    kwargs['all_page_name2id'] = all_page_name2id
    kwargs['max_df_normal_cnt'] = max_df_normal_cnt
    session_vector_list_abnormal, session_vector_len_list_abnormal, session_vector_label_list_abnormal, _, _ = prepare_abnormal_data(
        args, **kwargs)

    # 按照真实数据集的比例混合正常和异常数据，真实数据集的比例即 异常用户数 / 正常用户数
    # 这里假定session_vector_list和session_vector_list_abnormal就是真实数据集，因此不需要进行抽样，直接按照比例混合即可
    torch.manual_seed(args.random_seed)
    # 抽样正常数据
    normal_index = torch.randperm(session_vector_list_normal.shape[1])
    train_normal_index = normal_index[:int(session_vector_list_normal.shape[1] * args.train_ratio)]
    test_normal_index = normal_index[int(session_vector_list_normal.shape[1] * args.train_ratio):]
    train_normal_data = session_vector_list_normal[:, train_normal_index]
    train_normal_len = session_vector_len_list_normal[train_normal_index]
    train_normal_label = session_vector_label_list_normal[train_normal_index]
    test_normal_data = session_vector_list_normal[:, test_normal_index]
    test_normal_len = session_vector_len_list_normal[test_normal_index]
    test_normal_label = session_vector_label_list_normal[test_normal_index]
    # 抽样异常数据
    abnormal_index = torch.randperm(session_vector_list_abnormal.shape[1])
    train_abnormal_index = abnormal_index[:int(session_vector_list_abnormal.shape[1] * args.train_ratio)]
    test_abnormal_index = abnormal_index[int(session_vector_list_abnormal.shape[1] * args.train_ratio):]
    train_abnormal_data = session_vector_list_abnormal[:, train_abnormal_index]
    train_abnormal_len = session_vector_len_list_abnormal[train_abnormal_index]
    train_abnormal_label = session_vector_label_list_abnormal[train_abnormal_index]
    test_abnormal_data = session_vector_list_abnormal[:, test_abnormal_index]
    test_abnormal_len = session_vector_len_list_abnormal[test_abnormal_index]
    test_abnormal_label = session_vector_label_list_abnormal[test_abnormal_index]

    # 合并训练数据
    train_data = torch.cat((train_normal_data, train_abnormal_data), dim=1)
    train_len = torch.cat((train_normal_len, train_abnormal_len), dim=0)
    train_label = torch.cat((train_normal_label, train_abnormal_label), dim=0)

    # 合并测试数据
    test_data = torch.cat((test_normal_data, test_abnormal_data), dim=1)
    test_len = torch.cat((test_normal_len, test_abnormal_len), dim=0)
    test_label = torch.cat((test_normal_label, test_abnormal_label), dim=0)

    return train_data, train_len, train_label, test_data, test_len, test_label

def build_dataloader(args, **kwargs):
    """构建dataloader

    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数
    Returns:
        train_loader (torch.utils.data.DataLoader): 训练集的dataloader
        test_loader (torch.utils.data.DataLoader): 测试集的dataloader
    """
    train_data, train_len, train_label, test_data, test_len, test_label = prepare_data(
        args, **kwargs)
    train_set = MyDataset(args, train_data, train_label)
    test_set = MyDataset(args, test_data, test_label)
    train_loader = DataLoader(
        train_set,
        worker_init_fn=worker_init_fn_seed,
        batch_sampler=BalancedBatchSampler(args, train_set),
        **kwargs
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn_seed,
        **kwargs
    )
    return train_loader, test_loader


if __name__ == '__main__':
    pass
