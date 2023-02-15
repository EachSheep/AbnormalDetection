"""数据加载器
"""
import os
import pandas as pd
import torch
import json
from torch.utils.data import DataLoader

from dataloaders.MyDataset import MyDataset
from dataloaders.utlis import worker_init_fn_seed, BalancedBatchSampler, RandomedBatchSampler
import dataloaders.data2matrix_by_pagesession as d2mbps
import dataloaders.data2matrix_by_pageuser as d2mbpu
import dataloaders.data2matrix_by_worduser as d2mbwu


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
        valid_data (torch.LongTensor): 测试集的session向量
        valid_len (torch.LongTensor): 测试集的session向量长度
        valid_sid (pd.DataFrame): 测试集的session_id
        valid_uid (pd.DataFrame): 测试集的user_id
        valid_label (torch.LongTensor): 测试集的session向量标签
    """
    if args.use_cache:
        train_data = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'train_data.pkl'))
        train_len = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'train_len.pkl'))
        train_sid = pd.read_csv(os.path.join(
            args.experiment_dir, 'cache', f'train_sid.csv'))
        train_uid = pd.read_csv(os.path.join(
            args.experiment_dir, 'cache', f'train_uid.csv'))
        train_label = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'train_label.pkl'))

        valid_data = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'valid_data.pkl'))
        valid_len = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'valid_len.pkl'))
        valid_sid = pd.read_csv(os.path.join(
            args.experiment_dir, 'cache', f'valid_sid.csv'))
        valid_uid = pd.read_csv(os.path.join(
            args.experiment_dir, 'cache', f'valid_uid.csv'))
        valid_label = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'valid_label.pkl'))
        
        train_sid = train_sid['session_id'].values
        train_uid = train_uid['user_id'].values

        valid_sid = valid_sid['session_id'].values
        valid_uid = valid_uid['user_id'].values
        
    else:
        if args.data_type == 'pagesession':
            data_normal, len_normal, sid_normal, uid_normal, label_normal, \
                unknown_page_name_normal, unknown_page_len_normal = d2mbps.prepare_normal_data(
                    args, **kwargs)
            data_abnormal, len_abnormal, sid_abnormal, uid_abnormal, label_abnormal, \
                unknown_page_name_abnormal, unknown_page_len_abnormal = d2mbps.prepare_abnormal_data(
                    args, **kwargs)
        elif args.data_type == 'pageuser':
            data_normal, len_normal, sid_normal, uid_normal, label_normal, \
                unknown_page_name_normal, unknown_page_len_normal = d2mbpu.prepare_normal_data(
                    args, **kwargs)
            data_abnormal, len_abnormal, sid_abnormal, uid_abnormal, label_abnormal, \
                unknown_page_name_abnormal, unknown_page_len_abnormal = d2mbpu.prepare_abnormal_data(
                    args, **kwargs)
        elif args.data_type == 'worduser':
            data_normal, len_normal, sid_normal, uid_normal, label_normal, \
                unknown_page_name_normal, unknown_page_len_normal = d2mbwu.prepare_normal_data(
                    args, **kwargs)
            data_abnormal, len_abnormal, sid_abnormal, uid_abnormal, label_abnormal, \
                unknown_page_name_abnormal, unknown_page_len_abnormal = d2mbwu.prepare_abnormal_data(
                    args, **kwargs)
        else:
            raise ValueError(
                'args.data_type must be pagesession or pageuser or worduser')

        if not os.path.exists(os.path.join(args.experiment_dir, 'cache')):
            os.makedirs(os.path.join(args.experiment_dir, 'cache'))
        json.dump(unknown_page_name_normal, open(os.path.join(args.experiment_dir,
                  'cache', f'unknown_page_name_normal-train.json'), 'w'), indent=4)
        json.dump(unknown_page_len_normal, open(os.path.join(args.experiment_dir,
                  'cache', f'unknown_page_len_normal-train.json'), 'w'), indent=4)
        json.dump(unknown_page_name_abnormal, open(os.path.join(args.experiment_dir,
                                                                'cache', f'unknown_page_name_abnormal-train.json'), 'w'), indent=4)
        json.dump(unknown_page_len_abnormal, open(os.path.join(args.experiment_dir,
                                                               'cache', f'unknown_page_len_abnormal-train.json'), 'w'), indent=4)

        # 按照真实数据集的比例混合正常和异常数据，真实数据集的比例即 异常用户数 / 正常用户数
        # 这里假定session_vector_list和session_vector_list_abnormal就是真实数据集，因此不需要进行抽样，直接按照比例混合即可
        torch.manual_seed(args.random_seed)

        # 抽样正常数据
        normal_index = torch.randperm(data_normal.shape[0])
        train_normal_index = normal_index[:int(
            data_normal.shape[0] * args.train_ratio)]
        valid_normal_index = normal_index[int(
            data_normal.shape[0] * args.train_ratio):]
        train_data_normal = data_normal[train_normal_index]
        train_len_normal = len_normal[train_normal_index]
        train_sid_normal = sid_normal.iloc[train_normal_index.tolist(), :]
        train_uid_normal = uid_normal.iloc[train_normal_index.tolist(
        ), :]
        train_label_normal = label_normal[train_normal_index]
        valid_data_normal = data_normal[valid_normal_index]
        valid_len_normal = len_normal[valid_normal_index]
        valid_sid_normal = sid_normal.iloc[valid_normal_index.tolist(), :]
        valid_uid_normal = uid_normal.iloc[valid_normal_index.tolist(
        ), :]
        valid_label_normal = label_normal[valid_normal_index]

        # 抽样异常数据
        abnormal_index = torch.randperm(data_abnormal.shape[0])
        train_abnormal_index = abnormal_index[:int(
            data_abnormal.shape[0] * args.train_ratio)]
        valid_abnormal_index = abnormal_index[int(
            data_abnormal.shape[0] * args.train_ratio):]
        train_data_abnormal = data_abnormal[train_abnormal_index]
        train_len_abnormal = len_abnormal[train_abnormal_index]
        train_sid_abnormal = sid_abnormal.iloc[train_abnormal_index.tolist(
        ), :]
        train_uid_abnormal = uid_abnormal.iloc[train_abnormal_index.tolist(
        ), :]
        train_label_abnormal = label_abnormal[train_abnormal_index]
        valid_data_abnormal = data_abnormal[valid_abnormal_index]
        valid_len_abnormal = len_abnormal[valid_abnormal_index]
        valid_sid_abnormal = sid_abnormal.iloc[valid_abnormal_index.tolist(
        ), :]
        valid_uid_abnormal = uid_abnormal.iloc[valid_abnormal_index.tolist(
        ), :]
        valid_label_abnormal = label_abnormal[valid_abnormal_index]

        # 合并训练数据
        train_data = torch.cat((train_data_normal, train_data_abnormal), dim=0)
        train_len = torch.cat((train_len_normal, train_len_abnormal), dim=0)
        train_sid = pd.concat((train_sid_normal, train_sid_abnormal), axis=0)
        train_uid = pd.concat(
            (train_uid_normal, train_uid_abnormal), axis=0)
        train_label = torch.cat(
            (train_label_normal, train_label_abnormal), dim=0)
        train_sid = train_sid.reset_index(drop=True)
        train_uid = train_uid.reset_index(drop=True)

        # 合并测试数据
        valid_data = torch.cat((valid_data_normal, valid_data_abnormal), dim=0)
        valid_len = torch.cat((valid_len_normal, valid_len_abnormal), dim=0)
        valid_sid = pd.concat((valid_sid_normal, valid_sid_abnormal), axis=0)
        valid_uid = pd.concat(
            (valid_uid_normal, valid_uid_abnormal), axis=0)
        valid_label = torch.cat(
            (valid_label_normal, valid_label_abnormal), dim=0)
        valid_sid = valid_sid.reset_index(drop=True)
        valid_uid = valid_uid.reset_index(drop=True)

        if not os.path.exists(os.path.join(args.experiment_dir, 'cache')):
            os.makedirs(os.path.join(args.experiment_dir, 'cache'))
        torch.save(train_data, os.path.join(
            args.experiment_dir, 'cache', f'train_data.pkl'))
        torch.save(train_len, os.path.join(
            args.experiment_dir, 'cache', f'train_len.pkl'))
        train_sid.to_csv(os.path.join(
            args.experiment_dir, 'cache', f'train_sid.csv'), index=False)
        train_uid.to_csv(os.path.join(
            args.experiment_dir, 'cache', f'train_uid.csv'), index=False)
        torch.save(train_label, os.path.join(
            args.experiment_dir, 'cache', f'train_label.pkl'))

        torch.save(valid_data, os.path.join(
            args.experiment_dir, 'cache', f'valid_data.pkl'))
        torch.save(valid_len, os.path.join(
            args.experiment_dir, 'cache', f'valid_len.pkl'))
        valid_sid.to_csv(os.path.join(
            args.experiment_dir, 'cache', f'valid_sid.csv'), index=False)
        valid_uid.to_csv(os.path.join(
            args.experiment_dir, 'cache', f'valid_uid.csv'), index=False)
        torch.save(valid_label, os.path.join(
            args.experiment_dir, 'cache', f'valid_label.pkl'))

        train_sid = train_sid['session_id'].values
        train_uid = train_uid['user_id'].values

        valid_sid = valid_sid['session_id'].values
        valid_uid = valid_uid['user_id'].values

    return train_data, train_len, train_sid, train_uid, train_label, valid_data, valid_len, valid_sid, valid_uid, valid_label


def build_train_dataloader(args, **kwargs):
    """构建dataloader

    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数
    Returns:
        train_loader (torch.utils.data.DataLoader): 训练集的dataloader
        valid_loader (torch.utils.data.DataLoader): 测试集的dataloader
    """
    train_data, train_len, train_sid, train_uid, train_label, valid_data, valid_len, valid_sid, valid_uid, valid_label = prepare_train_data(
        args, **kwargs)
    # 查看数据的基本类型和shape
    print("train_data: ", train_data.shape, train_data.dtype)
    print("train_len: ", train_len.shape, train_len.dtype)
    print("train_sid: ", train_sid.shape, train_sid.dtype)
    print("train_uid: ", train_uid.shape, train_uid.dtype)
    print("train_label: ", train_label.shape, train_label.dtype)
    print("valid_data: ", valid_data.shape, valid_data.dtype)
    print("valid_len: ", valid_len.shape, valid_len.dtype)
    print("valid_sid: ", valid_sid.shape, valid_sid.dtype)
    print("valid_uid: ", valid_uid.shape, valid_uid.dtype)
    print("valid_label: ", valid_label.shape, valid_label.dtype)
    input()
    train_set = MyDataset(args, train_data, train_label,
                          **{"valid_lens": train_len, "sid": train_sid, "uid": train_uid})
    valid_set = MyDataset(args, valid_data, valid_label, **
                          {"valid_lens": valid_len, "sid": valid_sid, "uid": valid_uid})
    args.train_normal_num = train_set.normal_num()
    args.train_abnormal_num = train_set.abnormal_num()
    args.valid_normal_num = valid_set.normal_num()
    args.valid_abnormal_num = valid_set.abnormal_num()

    train_loader = DataLoader(
        train_set,
        worker_init_fn=worker_init_fn_seed,
        batch_sampler=BalancedBatchSampler(args, train_set),
        **kwargs
    )
    valid_loader = DataLoader(
        valid_set,
        worker_init_fn=worker_init_fn_seed,
        batch_sampler=RandomedBatchSampler(args, valid_set),
        **kwargs
    )
    return train_loader, valid_loader

def prepare_valid_data(args, **kwargs):
    """载入验证集的数据，验证集的数据应该在prepare_train_data中已经保存到cache中

    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数
    Returns:
        valid_data (torch.LongTensor): 测试集的session向量
        valid_len (torch.LongTensor): 测试集的session向量长度
        valid_sid (pd.DataFrame): 测试集的session_id
        valid_uid (pd.DataFrame): 测试集的user_id
        valid_label (torch.LongTensor): 测试集的session向量标签
    """
    valid_data = torch.load(os.path.join(
        args.experiment_dir, 'cache', f'valid_data.pkl'))
    valid_len = torch.load(os.path.join(
        args.experiment_dir, 'cache', f'valid_len.pkl'))
    valid_sid = pd.read_csv(os.path.join(
        args.experiment_dir, 'cache', f'valid_sid.csv'))
    valid_uid = pd.read_csv(os.path.join(
        args.experiment_dir, 'cache', f'valid_uid.csv'))
    valid_label = torch.load(os.path.join(
        args.experiment_dir, 'cache', f'valid_label.pkl'))
    
    valid_sid = valid_sid['session_id'].values
    valid_uid = valid_uid['user_id'].values

    return valid_data, valid_len, valid_sid, valid_uid, valid_label

def build_valid_dataloader(args, **kwargs):
    """构建dataloader

    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数
    Returns:
        valid_loader (torch.utils.data.DataLoader): 测试集的dataloader
    """
    valid_data, valid_len, valid_sid, valid_uid, valid_label = prepare_valid_data(
        args, **kwargs)
    valid_set = MyDataset(args, valid_data, valid_label, **
                         {"valid_lens": valid_len, "sid": valid_sid, "uid": valid_uid})
    args.valid_normal_num = valid_set.normal_num()
    args.valid_abnormal_num = valid_set.abnormal_num()
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn_seed,
        **kwargs
    )
    return valid_loader

def prepare_test_data(args, **kwargs):
    """载入数据，划分训练集和测试集，划分的依据是args.train_ratio

    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数
    Returns:
        test_data (torch.LongTensor): 测试集的session向量
        test_len (torch.LongTensor): 测试集的session向量长度
        test_sid (pd.DataFrame): 测试集的session_id
        test_uid (pd.DataFrame): 测试集的user_id
        test_label (torch.LongTensor): 测试集的session向量标签
    """
    if args.use_cache:
        test_data = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'test_data.pkl'))
        test_len = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'test_len.pkl'))
        test_sid = pd.read_csv(os.path.join(
            args.experiment_dir, 'cache', f'test_sid.csv'))
        test_uid = pd.read_csv(os.path.join(
            args.experiment_dir, 'cache', f'test_uid.csv'))
        test_label = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'test_label.pkl'))
    else:
        if args.data_type == "pagesession":
            data_normal, len_normal, sid_normal, uid_normal, label_normal, unknown_page_name_normal, unknown_page_len_normal = d2mbps.prepare_normal_data(
                args, **kwargs)
            data_abnormal, len_abnormal, sid_abnormal, uid_abnormal, label_abnormal, unknown_page_name_abnormal, unknown_page_len_abnormal = d2mbps.prepare_abnormal_data(
                args, **kwargs)
        elif args.data_type == "pageuser":
            data_normal, len_normal, sid_normal, uid_normal, label_normal, unknown_page_name_normal, unknown_page_len_normal = d2mbpu.prepare_normal_data(
                args, **kwargs)
            data_abnormal, len_abnormal, sid_abnormal, uid_abnormal, label_abnormal, unknown_page_name_abnormal, unknown_page_len_abnormal = d2mbpu.prepare_abnormal_data(
                args, **kwargs)
        elif args.data_type == 'worduser':
            data_normal, len_normal, sid_normal, uid_normal, label_normal, unknown_page_name_normal, unknown_page_len_normal = d2mbwu.prepare_normal_data(
                args, **kwargs)
            data_abnormal, len_abnormal, sid_abnormal, uid_abnormal, label_abnormal, unknown_page_name_abnormal, unknown_page_len_abnormal = d2mbwu.prepare_abnormal_data(
                args, **kwargs)
        else:
            raise ValueError(
                "args.data_type must be pagesession or pageuser or worduser")

        if not os.path.exists(os.path.join(args.experiment_dir, 'cache')):
            os.makedirs(os.path.join(args.experiment_dir, 'cache'))
        json.dump(unknown_page_name_normal, open(os.path.join(args.experiment_dir,
                                                              'cache', f'unknown_page_name_normal-test.json'), 'w'), indent=4)
        json.dump(unknown_page_len_normal, open(os.path.join(args.experiment_dir,
                                                             'cache', f'unknown_page_len_normal-test.json'), 'w'), indent=4)
        json.dump(unknown_page_name_abnormal, open(os.path.join(args.experiment_dir,
                                                                'cache', f'unknown_page_name_abnormal-test.json'), 'w'), indent=4)
        json.dump(unknown_page_len_abnormal, open(os.path.join(args.experiment_dir,
                                                               'cache', f'unknown_page_len_abnormal-test.json'), 'w'), indent=4)

        # 合并训练数据
        test_data = torch.cat((data_normal, data_abnormal), dim=0)
        test_len = torch.cat((len_normal, len_abnormal), dim=0)
        test_sid = pd.concat((sid_normal, sid_abnormal), axis=0)
        test_uid = pd.concat((uid_normal, uid_abnormal), axis=0)
        test_label = torch.cat((label_normal, label_abnormal), dim=0)
        test_sid = test_sid.reset_index(drop=True)
        test_uid = test_uid.reset_index(drop=True)

        torch.save(test_data, os.path.join(
            args.experiment_dir, 'cache', f'test_data.pkl'))
        torch.save(test_len, os.path.join(
            args.experiment_dir, 'cache', f'test_len.pkl'))
        test_sid.to_csv(os.path.join(
            args.experiment_dir, 'cache', f'test_sid.csv'), index=False)
        test_uid.to_csv(os.path.join(
            args.experiment_dir, 'cache', f'test_uid.csv'), index=False)
        torch.save(test_label, os.path.join(
            args.experiment_dir, 'cache', f'test_label.pkl'))

        test_sid = test_sid['session_id'].values
        test_uid = test_uid['user_id'].values

    return test_data, test_len, test_sid, test_uid, test_label


def build_test_dataloader(args, **kwargs):
    """构建dataloader

    Args:
        args (argparse.ArgumentParser()): 参数
        kwargs (dict): 参数
    Returns:
        test_loader (torch.utils.data.DataLoader): 测试集的dataloader
    """
    test_data, test_len, test_sid, test_uid, test_label = prepare_test_data(
        args, **kwargs)
    test_set = MyDataset(args, test_data, test_label, **
                         {"valid_lens": test_len, "sid": test_sid, "uid": test_uid})
    args.test_normal_num = test_set.normal_num()
    args.test_abnormal_num = test_set.abnormal_num()
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
