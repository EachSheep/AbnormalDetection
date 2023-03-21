import os
import torch
from torch.utils.tensorboard import SummaryWriter
import json
import time

from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
import numpy as np

from Trainer import Trainer
from Tester import Tester
from argparser import args
from utils.set_logger import set_logger

if __name__ == '__main__':

    torch.manual_seed(args.random_seed)
    if args.cache_dir == '':
        runs_dir = os.path.join(args.experiment_dir, 'runs')
    else:
        runs_dir = os.path.join(args.cache_dir, 'runs')
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    summarywriter_dir = os.path.join(runs_dir, 'train')
    if os.path.exists(summarywriter_dir):
        for root, dirs, files in os.walk(summarywriter_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(summarywriter_dir)
        os.makedirs(summarywriter_dir)
    else:
        os.makedirs(summarywriter_dir)

    logger = set_logger(summarywriter_dir)
    # writer = SummaryWriter(summarywriter_dir)
    trainer = Trainer(args)

    begin_time = time.time()
    for cur_epoch in range(0, trainer.args.epochs):
        cur_train_loss = trainer.train(cur_epoch)
        logger.info("TRAIN LOSS: %.4f" % (cur_train_loss))
        # writer.add_scalars("mini-valid/loss", {"train" : cur_train_loss}, cur_epoch)
        
        # if (cur_epoch + 1) % 5 == 0:
        #     cur_roc, cur_pr, precision, recall, F1, cur_test_loss, cur_label, cur_predict, total_uid  = trainer.eval(cur_epoch)
        #     logger.info("ROC-AUC: %.4f, PR-AUC: %.4f, PRECISION: %.4f, RECALL: %.4f, F1: %.4f, VALID LOSS: %.4f" % (cur_roc, cur_pr, precision, recall, F1, cur_test_loss))
        #     writer.add_scalars("mini-valid/loss", {"loss_valid" : cur_train_loss}, cur_epoch)
        #     writer.add_scalar("mini-valid/roc", cur_roc, cur_epoch)
        #     writer.add_scalar("mini-valid/pr", cur_pr, cur_epoch)
        #     writer.add_scalar("mini-valid/precision", precision, cur_epoch)
        #     writer.add_scalar("mini-valid/recall", recall, cur_epoch)
        #     writer.add_scalar("mini-valid/F1", F1, cur_epoch)
        #     writer.add_pr_curve(f"mini-valid/pr_curve-{cur_epoch}", cur_label, cur_predict, global_step=cur_epoch)
    
    # writer.flush()
    # writer.close()
    if args.cache_dir == '':
        model_path = os.path.join(args.experiment_dir, 'models', args.weight_name)
    else:
        model_path = os.path.join(args.cache_dir, 'models', args.weight_name)
    trainer.save_weights(model_path = model_path)
    end_time = time.time()
    print("Training time: {:.3f}".format((end_time - begin_time))) # 113.505 / 856408 = 0.00013
    args.train_time_per_epoch = "{:.3f}".format((end_time - begin_time) / args.epochs)
    # cur_roc, cur_pr, precision, recall, F1, cur_test_loss, cur_label, cur_predict, total_uid = trainer.test()
    # np.save(os.path.join(summarywriter_dir, 'valid_label.npy'), cur_label)
    # np.save(os.path.join(summarywriter_dir, 'valid_predict.npy'), cur_predict)
    # np.save(os.path.join(summarywriter_dir, 'valid_uid.npy'), total_uid)
    # logger.info("ROC-AUC: %.4f, PR-AUC: %.4f, PRECISION: %.4f, RECALL: %.4f, F1: %.4f, VALID LOSS: %.4f" % (cur_roc, cur_pr, precision, recall, F1, cur_test_loss))
    
    # args.ROCAUC = cur_roc
    # args.PRAUC = cur_pr
    # args.PRECISION = precision
    # args.RECALL = recall
    # args.F1 = F1
    # args.VALID_LOSS = cur_test_loss
    if args.cache_dir == '':
        preserve_dir = os.path.join(args.experiment_dir, f'{args.log_dir}-{args.log_label}')
    else:
        preserve_dir = os.path.join(args.cache_dir, f'{args.log_dir}-{args.log_label}')
    if not os.path.exists(preserve_dir):
        os.makedirs(preserve_dir)
    setting_path = os.path.join(preserve_dir, 'train_setting.json')
    json.dump(args.__dict__, open(setting_path, 'w'), indent=4)

    # valid
    # tester = Tester(args)
    # cur_roc, cur_pr, precision, recall, F1, cur_test_loss, cur_label, cur_predict, total_uid = tester.eval(state_dict_path = model_path)
    # logger.info("ROC-AUC: %.4f, PR-AUC: %.4f, PRECISION: %.4f, RECALL: %.4f, F1: %.4f, VALID LOSS: %.4f" % (cur_roc, cur_pr, precision, recall, F1, cur_test_loss))
    # np.save(os.path.join(summarywriter_dir, 'valid_label.npy'), cur_label)
    # np.save(os.path.join(summarywriter_dir, 'valid_predict.npy'), cur_predict)
    # np.save(os.path.join(summarywriter_dir, 'valid_uid.npy'), total_uid)
    # preserve_dir = os.path.join(args.experiment_dir, f'{args.log_dir}-{args.log_label}')
    # if not os.path.exists(preserve_dir):
    #     os.makedirs(preserve_dir)
    # args.ROCAUC = cur_roc
    # args.PRAUC = cur_pr
    # args.PRECISION = precision
    # args.RECALL = recall
    # args.F1 = F1
    # args.VALID_LOSS = cur_test_loss
    # setting_path = os.path.join(preserve_dir, 'train_setting.json')
    # json.dump(args.__dict__, open(setting_path, 'w'), indent=4)

    # 大概15个epoch收敛
    # 训练全部的大概: 
    # 测试:444.0410s,(1165655 + 30855)

    # devnet 5个epoch收敛: 总共耗时:10851.901(12个epoch),因此耗时:10851.901/12*5 = 4521.625

    # 新数据集的结果
    
    # BCE, 随机采样
    # after_newtest_data :  88.53 39.23
    # after_newtest_data :  88.53 59.40
    # after_newtest_data :  88.53 55.48
    # after_newtest_data :  88.53 52.78
    
    # # BCE, 平衡采样
    # after_newtest_data :  89.01 39.33
    # after_newtest_data :  89.01 56.20
    # after_newtest_data :  89.01 54.10
    # after_newtest_data :  89.01 51.94

    # WeightedBCE, 随机采样
    # # 1781.497 2 epochs
    # 0.1, 0.9的权重
    # after_newtest_data :  87.50 37.97
    # after_newtest_data :  87.50 61.10
    # after_newtest_data :  87.50 56.84
    # after_newtest_data :  87.50 51.63

    # WeightedBCE, 随机采样
    # # 1781.497 2 epochs
    # 0.01, 0.99的权重
    # after_newtest_data :  87.40 37.17
    # after_newtest_data :  87.40 66.40
    # after_newtest_data :  87.40 56.62
    # after_newtest_data :  87.40 51.78
    

    # WeightedBCE, 平衡采样
    # # 1780.615 2 epochs
    # 0.1, 0.9的权重
    # after_newtest_data :  88.81 39.51
    # after_newtest_data :  88.81 63.30
    # after_newtest_data :  88.81 58.00
    # after_newtest_data :  88.81 53.35

    # # 0.01, 0.99的权重
    # after_newtest_data :  88.62 39.25
    # after_newtest_data :  88.62 74.70
    # after_newtest_data :  88.62 58.10
    # after_newtest_data :  88.62 53.20

    # 0.001, 0.999的权重
    # after_newtest_data :  88.29 39.08
    # after_newtest_data :  88.29 74.00
    # after_newtest_data :  88.29 59.24
    # after_newtest_data :  88.29 53.58

    # # # 0.01, 0.99的权重，部分训练
    # after_newtest_data :  87.16 36.56
    # after_newtest_data :  87.16 58.10
    # after_newtest_data :  87.16 54.62
    # after_newtest_data :  87.16 51.68

    # 对比学习+WeightedBCE, 平衡采样
    # 896.305, 1 epoch
    # after_newtest_data :  88.53 39.03
    # after_newtest_data :  88.53 53.60
    # after_newtest_data :  88.53 52.18
    # after_newtest_data :  88.53 51.15

    # DevNet, 平衡采样
    # after_newtest_data :  85.73 32.06
    # after_newtest_data :  85.73 41.00
    # after_newtest_data :  85.73 46.28
    # after_newtest_data :  85.73 47.92
    
    # # DevNet, 随机采样
    # after_newtest_data :  62.28 6.04
    # after_newtest_data :  62.28 10.40
    # after_newtest_data :  62.28 6.78
    # after_newtest_data :  62.28 6.33