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
        test_data (torch.LongTensor): 测试集的session向量
        test_len (torch.LongTensor): 测试集的session向量长度
        test_id (pd.DataFrame): 测试集的session_id
        test_uid (pd.DataFrame): 测试集的user_id
        test_label (torch.LongTensor): 测试集的session向量标签
    """
    if args.use_cache:
        
        feature_normal = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'feature_normal.pkl'))
        feature_len_normal = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'feature_len_normal.pkl'))
        feature_sid_normal = pd.read_csv(os.path.join(
            args.experiment_dir, 'cache', f'feature_sid_normal.csv'))
        feature_uid_normal = pd.read_csv(os.path.join(
            args.experiment_dir, 'cache', f'feature_uid_normal.csv'))
        feature_label_normal = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'feature_label_normal.pkl'))
        unknown_page_name_normal = json.load(open(os.path.join(args.experiment_dir,
                    'cache', f'unknown_page_name_normal.json'), 'r'))
        unknown_page_len_normal = json.load(open(os.path.join(args.experiment_dir,
                    'cache', f'unknown_page_len_normal.json'), 'r'))
        
        feature_abnormal = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'feature_abnormal.pkl'))
        feature_len_abnormal = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'feature_len_abnormal.pkl'))
        feature_sid_abnormal = pd.read_csv(os.path.join(
            args.experiment_dir, 'cache', f'feature_sid_abnormal.csv'))
        feature_uid_abnormal = pd.read_csv(os.path.join(
            args.experiment_dir, 'cache', f'feature_uid_abnormal.csv'))
        feature_label_abnormal = torch.load(os.path.join(
            args.experiment_dir, 'cache', f'feature_label_abnormal.pkl'))
        unknown_page_name_abnormal = json.load(open(os.path.join(args.experiment_dir,
                    'cache', f'unknown_page_name_abnormal.json'), 'r'))
        unknown_page_len_abnormal = json.load(open(os.path.join(args.experiment_dir,
                    'cache', f'unknown_page_len_abnormal.json'), 'r'))
    else:
        if args.data_type == 'pagesession':
            feature_normal, feature_len_normal, feature_sid_normal, feature_uid_normal, feature_label_normal, \
                unknown_page_name_normal, unknown_page_len_normal = d2mbps.prepare_normal_data(
                    args, **kwargs)
        elif args.data_type == 'pageuser':
            feature_normal, feature_len_normal, feature_sid_normal, feature_uid_normal, feature_label_normal, \
                unknown_page_name_normal, unknown_page_len_normal = d2mbpu.prepare_normal_data(
                    args, **kwargs)
        elif args.data_type == 'worduser':
            feature_normal, feature_len_normal, feature_sid_normal, feature_uid_normal, feature_label_normal, \
                unknown_page_name_normal, unknown_page_len_normal = d2mbwu.prepare_normal_data(
                    args, **kwargs)
        else:
            raise ValueError('args.data_type must be pagesession or pageuser')
        if not os.path.exists(os.path.join(args.experiment_dir, 'cache')):
            os.makedirs(os.path.join(args.experiment_dir, 'cache'))
        torch.save(feature_normal, os.path.join(
            args.experiment_dir, 'cache', f'feature_normal.pkl'))
        torch.save(feature_len_normal, os.path.join(
            args.experiment_dir, 'cache', f'feature_len_normal.pkl'))
        feature_sid_normal.to_csv(os.path.join(
            args.experiment_dir, 'cache', f'feature_sid_normal.csv'), index=False)
        feature_uid_normal.to_csv(os.path.join(
            args.experiment_dir, 'cache', f'feature_uid_normal.csv'), index=False)
        torch.save(feature_label_normal, os.path.join(
            args.experiment_dir, 'cache', f'feature_label_normal.pkl'))
        json.dump(unknown_page_name_normal, open(os.path.join(args.experiment_dir,
                  'cache', f'unknown_page_name_normal.json'), 'w'), indent=4)
        json.dump(unknown_page_len_normal, open(os.path.join(args.experiment_dir,
                  'cache', f'unknown_page_len_normal.json'), 'w'), indent=4)
        
        if args.data_type == 'pagesession':
            feature_abnormal, feature_len_abnormal, feature_sid_abnormal, feature_uid_abnormal, feature_label_abnormal, \
                unknown_page_name_abnormal, unknown_page_len_abnormal = d2mbps.prepare_abnormal_data(
                    args, **kwargs)
        elif args.data_type == 'pageuser':
            feature_abnormal, feature_len_abnormal, feature_sid_abnormal, feature_uid_abnormal, feature_label_abnormal, \
                unknown_page_name_abnormal, unknown_page_len_abnormal = d2mbpu.prepare_abnormal_data(
                    args, **kwargs)
        elif args.data_type == 'worduser':
            feature_abnormal, feature_len_abnormal, feature_sid_abnormal, feature_uid_abnormal, feature_label_abnormal, \
                unknown_page_name_abnormal, unknown_page_len_abnormal = d2mbwu.prepare_abnormal_data(
                    args, **kwargs)
        else:
            raise ValueError('args.data_type must be pagesession or pageuser')
        torch.save(feature_abnormal, os.path.join(
            args.experiment_dir, 'cache', f'feature_abnormal.pkl'))
        torch.save(feature_len_abnormal, os.path.join(
            args.experiment_dir, 'cache', f'feature_len_abnormal.pkl'))
        feature_sid_abnormal.to_csv(os.path.join(
            args.experiment_dir, 'cache', f'feature_sid_abnormal.csv'), index=False)
        feature_uid_abnormal.to_csv(os.path.join(
            args.experiment_dir, 'cache', f'feature_uid_abnormal.csv'), index=False)
        torch.save(feature_label_abnormal, os.path.join(
            args.experiment_dir, 'cache', f'feature_label_abnormal.pkl'))
        json.dump(unknown_page_name_abnormal, open(os.path.join(args.experiment_dir,
                    'cache', f'unknown_page_name_abnormal.json'), 'w'), indent=4)
        json.dump(unknown_page_len_abnormal, open(os.path.join(args.experiment_dir,
                    'cache', f'unknown_page_len_abnormal.json'), 'w'), indent=4)
    unknown_page_name_normal.update(unknown_page_name_abnormal)
    unknown_page_len_normal.update(unknown_page_len_abnormal)
    unknown_page_name = unknown_page_name_normal
    unknown_page_len = unknown_page_len_normal
    del unknown_page_name_normal, unknown_page_len_normal, unknown_page_name_abnormal, unknown_page_len_abnormal

    if not os.path.exists(os.path.join(args.experiment_dir, 'jsons')):
        os.makedirs(os.path.join(args.experiment_dir, 'jsons'))
    json.dump(unknown_page_name, open(os.path.join(args.experiment_dir,
              'jsons', f'unknown_page_name-train.json'), 'w'), indent=4)
    json.dump(unknown_page_len, open(os.path.join(args.experiment_dir,
              'jsons', f'unknown_page_len-train.json'), 'w'), indent=4)

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
                          **{"valid_lens": train_len, "sid": train_sid, "uid": train_uid})
    test_set = MyDataset(args, test_data, test_label, **
                         {"valid_lens": test_len, "sid": test_id, "uid": test_uid})
    args.train_normal_num = train_set.normal_num()
    args.train_abnormal_num = train_set.abnormal_num()
    args.test_normal_num = test_set.normal_num()
    args.test_abnormal_num = test_set.abnormal_num()

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
        raise ValueError("args.data_type must be pagesession or pageuser or worduser")
    unknown_page_name_normal.update(unknown_page_name_abnormal)
    unknown_page_len_normal.update(unknown_page_len_abnormal)
    unknown_page_name = unknown_page_name_normal
    unknown_page_len = unknown_page_len_normal
    del unknown_page_name_normal, unknown_page_len_normal, unknown_page_name_abnormal, unknown_page_len_abnormal

    if not os.path.exists(os.path.join(args.experiment_dir, 'jsons')):
        os.makedirs(os.path.join(args.experiment_dir, 'jsons'))
    json.dump(unknown_page_name, open(os.path.join(args.experiment_dir,
              'jsons', f'unknown_page_name-test.json'), 'w'), indent=4)
    json.dump(unknown_page_len, open(os.path.join(args.experiment_dir,
              'jsons', f'unknown_page_len-test.json'), 'w'), indent=4)

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
                         {"valid_lens": test_len, "sid": test_id, "uid": test_uid})
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
