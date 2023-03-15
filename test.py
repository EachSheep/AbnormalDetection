import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
import numpy as np

from Tester import Tester
from argparser import args
from utils.set_logger import set_logger


if __name__ == '__main__':

    torch.manual_seed(args.random_seed)

    if args.cache_dir == '':
        summarywriter_dir = os.path.join(args.experiment_dir, 'runs', 'train')
    else:
        summarywriter_dir = os.path.join(args.cache_dir, 'runs', 'train')
    writer = SummaryWriter(summarywriter_dir)

    tester = Tester(args)
    logger = set_logger(summarywriter_dir)
    if args.cache_dir == '':
        model_path = os.path.join(args.experiment_dir, 'models', args.weight_name)
    else:
        model_path = os.path.join(args.cache_dir, 'models', args.weight_name)

    begin_time = time.time()
    # cur_roc, cur_pr, cur_test_loss, cur_label, cur_predict, total_uid = tester.eval(
    cur_roc, cur_pr, precision, recall, F1, cur_test_loss, cur_label, cur_predict, total_uid = tester.eval(
        state_dict_path=model_path)
    end_time = time.time()
    logger.info("Time: %.4f" % (end_time - begin_time))
    np.save(os.path.join(summarywriter_dir, 'valid_label.npy'), cur_label)
    np.save(os.path.join(summarywriter_dir, 'valid_predict.npy'), cur_predict)
    np.save(os.path.join(summarywriter_dir, 'valid_uid.npy'), total_uid)
    logger.info("ROC-AUC: %.4f, PR-AUC: %.4f, PRECISION: %.4f, RECALL: %.4f, F1: %.4f, VALID LOSS: %.4f" % (cur_roc, cur_pr, precision, recall, F1, cur_test_loss))

    writer.flush()
    writer.close()
