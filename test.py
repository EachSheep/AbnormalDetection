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

    cur_roc, cur_pr, cur_test_loss, cur_label, cur_predict, total_uid, total_sid = tester.eval(
        state_dict_path=model_path)
    np.save(os.path.join(summarywriter_dir, 'valid_label.npy'), cur_label)
    np.save(os.path.join(summarywriter_dir, 'valid_predict.npy'), cur_predict)
    np.save(os.path.join(summarywriter_dir, 'valid_uid.npy'), total_uid)
    np.save(os.path.join(summarywriter_dir, 'valid_sid.npy'), total_sid)

    # for i in range(1000, len(cur_label), 1000):
    #     label, predict = cur_label[:i], cur_predict[:i]
    #     try:
    #         cur_roc = roc_auc_score(label, predict)
    #         cur_pr = average_precision_score(label, predict)
    #         writer.add_scalar("test/roc", cur_roc, i)
    #         writer.add_scalar("test/pr", cur_pr, i)
    #         logger.info("i: %d, ROC-AUC: %.4f, PR-AUC: %.4f" % (i, cur_roc, cur_pr))
    #     except ValueError:
    #         # ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
    #         if len(np.unique(label)) == 1 and label[0] == 1:
    #             writer.add_scalar("test/ValueError", 1, i)
    #         else:
    #             writer.add_scalar("test/ValueError", 0, i)
    #         pass
    # writer.add_pr_curve(f"test/pr_curve-test-1000",
    #                     cur_label[:1000], cur_predict[:1000])

    writer.flush()
    writer.close()
