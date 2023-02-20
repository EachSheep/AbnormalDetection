import os
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
import numpy as np

from Tester import Tester
from argparser import args
from utils.set_logger import set_logger


if __name__ == '__main__':

    torch.manual_seed(args.random_seed)

    summarywriter_dir = os.path.join(args.experiment_dir, 'runs', 'train')
    writer = SummaryWriter(summarywriter_dir)

    tester = Tester(args)
    logger = set_logger(summarywriter_dir)
    model_path = os.path.join(args.experiment_dir, 'models', args.weight_name)

    cur_roc, cur_pr, cur_test_loss, cur_label, cur_predict, total_uid = tester.eval(
        state_dict_path=model_path)
    np.save(os.path.join(summarywriter_dir, 'valid_label.npy'), cur_label)
    np.save(os.path.join(summarywriter_dir, 'valid_predict.npy'), cur_predict)
    np.save(os.path.join(summarywriter_dir, 'valid_uid.npy'), total_uid)

    writer.flush()
    writer.close()
