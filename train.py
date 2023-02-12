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
    writer = SummaryWriter(summarywriter_dir)
    logger = set_logger(summarywriter_dir)
    
    trainer = Trainer(args)
    model_path = os.path.join(args.experiment_dir, 'models', args.weight_name)

    begin_time = time.time()
    for cur_epoch in range(0, trainer.args.epochs):
        cur_train_loss = trainer.train(cur_epoch)
        writer.add_scalars("mini-valid/loss", {"train" : cur_train_loss}, cur_epoch)
        
        if (cur_epoch + 1) % 5 == 0:
            cur_roc, cur_pr, cur_test_loss, cur_label, cur_predict, total_uid, total_sid  = trainer.eval(cur_epoch)
            logger.info("ROC-AUC: %.4f, PR-AUC: %.4f, VALID LOSS: %.4f" % (cur_roc, cur_pr, cur_test_loss))
            writer.add_scalars("mini-valid/loss", {"loss_valid" : cur_train_loss}, cur_epoch)
            writer.add_scalar("mini-valid/roc", cur_roc, cur_epoch)
            writer.add_scalar("mini-valid/pr", cur_pr, cur_epoch)
            writer.add_pr_curve(f"mini-valid/pr_curve-{cur_epoch}", cur_label, cur_predict, global_step=cur_epoch)

    end_time = time.time()
    args.train_time_per_epoch = "{:.3f}".format((end_time - begin_time) / args.epochs)
    setting_path = os.path.join(summarywriter_dir, 'train_setting.json')
    json.dump(args.__dict__, open(setting_path, 'w'), indent=4)
    trainer.save_weights(model_path = model_path)

    # valid
    tester = Tester(args)
    model_path = os.path.join(args.experiment_dir, 'models', args.weight_name)
    cur_roc, cur_pr, cur_test_loss, cur_label, cur_predict, total_uid, total_sid = tester.eval(state_dict_path = model_path)
    np.save(os.path.join(summarywriter_dir, 'valid_label.npy'), cur_label)
    np.save(os.path.join(summarywriter_dir, 'valid_predict.npy'), cur_predict)
    np.save(os.path.join(summarywriter_dir, 'valid_uid.npy'), total_uid)
    np.save(os.path.join(summarywriter_dir, 'valid_sid.npy'), total_sid)

    # for i in range(1000, len(cur_label), 1000):
    #     label, predict = cur_label[:i], cur_predict[:i]
    #     try:
    #         cur_roc = roc_auc_score(label, predict)
    #         cur_pr = average_precision_score(label, predict)
    #         writer.add_scalar("valid/roc", cur_roc, i)
    #         writer.add_scalar("valid/pr", cur_pr, i)
    #         logger.info("i: %d, ROC-AUC: %.4f, PR-AUC: %.4f" % (i, cur_roc, cur_pr))
    #     except ValueError:
    #         # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
    #         if len(np.unique(label)) == 1 and label[0] == 1:  
    #             writer.add_scalar("valid/ValueError", 1, i)
    #         else:
    #             writer.add_scalar("valid/ValueError", 0, i)

    # writer.add_pr_curve(f"valid/pr_curve-valid-100", cur_label[:100], cur_predict[:1000])

    writer.flush()
    writer.close()