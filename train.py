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
    runs_dir = os.path.join(args.experiment_dir, 'runs')
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
    # model_path = os.path.join(args.experiment_dir, 'models', args.weight_name)
    # trainer.save_weights(model_path = model_path)
    end_time = time.time()
    args.train_time_per_epoch = "{:.3f}".format((end_time - begin_time) / args.epochs)
    cur_roc, cur_pr, precision, recall, F1, cur_test_loss, cur_label, cur_predict, total_uid = trainer.test()
    # np.save(os.path.join(summarywriter_dir, 'valid_label.npy'), cur_label)
    # np.save(os.path.join(summarywriter_dir, 'valid_predict.npy'), cur_predict)
    # np.save(os.path.join(summarywriter_dir, 'valid_uid.npy'), total_uid)
    logger.info("ROC-AUC: %.4f, PR-AUC: %.4f, PRECISION: %.4f, RECALL: %.4f, F1: %.4f, VALID LOSS: %.4f" % (cur_roc, cur_pr, precision, recall, F1, cur_test_loss))
    preserve_dir = os.path.join(args.experiment_dir, f'{args.log_dir}-{args.log_label}')
    if not os.path.exists(preserve_dir):
        os.makedirs(preserve_dir)
    args.ROCAUC = cur_roc
    args.PRAUC = cur_pr
    args.PRECISION = precision
    args.RECALL = recall
    args.F1 = F1
    args.VALID_LOSS = cur_test_loss
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