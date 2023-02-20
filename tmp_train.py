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
    if not os.path.exists(summarywriter_dir):
        os.makedirs(summarywriter_dir)
    if os.path.exists(summarywriter_dir):
        for root, dirs, files in os.walk(summarywriter_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        # os.rmdir(summarywriter_dir)
        # os.makedirs(summarywriter_dir)
    logger = set_logger(summarywriter_dir)

    embedding_dim = [300, 340, 380, 420]
    ffn_num_hiddens = [160, 200, 240, 280, 320]
    num_heads = [2, 4, 8]
    num_layers = [2, 4]
    dropout = [0.5, 0.3]
    criterion = ['BCE']
    lr = [0.0002]
    steps_per_epoch = [40, 80, 120]
    batch_size = [64, 128, 256]

    total_tests = len(embedding_dim) * len(ffn_num_hiddens) * len(num_heads) * len(num_layers) * len(dropout) * len(criterion) * len(lr) * len(steps_per_epoch) * len(batch_size)
    logger.info(f"total tests: {total_tests}.")

    max_F1 = 0
    max_setting = None

    i = 0
    for cur_embedding_dim in embedding_dim:
        for cur_ffn_num_hiddens in ffn_num_hiddens:
            for cur_num_heads in num_heads:
                for cur_num_layers in num_layers:
                    for cur_dropout in dropout:
                        for cur_crit in criterion:
                            for cur_lr in lr:
                                for cur_steps_per_epoch in steps_per_epoch:
                                    for cur_batch_size in batch_size:
                                        args.embedding_dim = cur_embedding_dim
                                        args.ffn_num_hiddens = cur_ffn_num_hiddens
                                        args.num_heads = cur_num_heads
                                        args.num_layers = cur_num_layers
                                        args.dropout = cur_dropout
                                        args.criterion = cur_crit
                                        args.lr = cur_lr
                                        args.steps_per_epoch = cur_steps_per_epoch
                                        args.batch_size = cur_batch_size
                                        
                                        i += 1
                                        logger.info("test {} / {}:".format(i, total_tests))
                                        # writer = SummaryWriter(summarywriter_dir)

                                        trainer = Trainer(args)

                                        begin_time = time.time()
                                        for cur_epoch in range(0, trainer.args.epochs):
                                            cur_train_loss = trainer.train(cur_epoch)
                                            # writer.add_scalars("mini-valid/loss", {"train" : cur_train_loss}, cur_epoch)
                                            
                                            if (cur_epoch + 1) % 5 == 0:
                                                cur_roc, cur_pr, precision, recall, F1, cur_test_loss, cur_label, cur_predict, total_uid  = trainer.eval(cur_epoch)
                                                logger.info("ROC-AUC: %.4f, PR-AUC: %.4f, PRECISION: %.4f, RECALL: %.4f, F1: %.4f, VALID LOSS: %.4f" % (cur_roc, cur_pr, precision, recall, F1, cur_test_loss))
                                                # writer.add_scalars("mini-valid/loss", {"loss_valid" : cur_train_loss}, cur_epoch)
                                                # writer.add_scalar("mini-valid/roc", cur_roc, cur_epoch)
                                                # writer.add_scalar("mini-valid/pr", cur_pr, cur_epoch)
                                                # writer.add_scalar("mini-valid/precision", precision, cur_epoch)
                                                # writer.add_scalar("mini-valid/recall", recall, cur_epoch)
                                                # writer.add_scalar("mini-valid/F1", F1, cur_epoch)
                                                # writer.add_pr_curve(f"mini-valid/pr_curve-{cur_epoch}", cur_label, cur_predict, global_step=cur_epoch)
                                        
                                        # writer.flush()
                                        # writer.close()
                                        
                                        end_time = time.time()
                                        args.train_time_per_epoch = "{:.3f}".format((end_time - begin_time) / args.epochs)

                                        model_path = os.path.join(args.experiment_dir, 'models', args.weight_name.split('.')[0] + '_tmp' + '.' + args.weight_name.split('.')[1])
                                        trainer.save_weights(model_path = model_path)

                                        # valid
                                        tester = Tester(args)
                                        model_path = os.path.join(args.experiment_dir, 'models', args.weight_name.split('.')[0] + '_tmp' + '.' + args.weight_name.split('.')[1])
                                        cur_roc, cur_pr, precision, recall, F1, cur_test_loss, cur_label, cur_predict, total_uid = tester.eval(state_dict_path = model_path)
                                        logger.info("ROC-AUC: %.4f, PR-AUC: %.4f, PRECISION: %.4f, RECALL: %.4f, F1: %.4f, VALID LOSS: %.4f" % (cur_roc, cur_pr, precision, recall, F1, cur_test_loss))
                                        
                                        if F1 > max_F1:
                                            max_F1 = F1
                                            max_setting = args.__dict__
                                            np.save(os.path.join(summarywriter_dir, 'valid_label.npy'), cur_label)
                                            np.save(os.path.join(summarywriter_dir, 'valid_predict.npy'), cur_predict)
                                            np.save(os.path.join(summarywriter_dir, 'valid_uid.npy'), total_uid)
                                            logger.info(f"max F1: {max_F1} occurs at test {i}." )

                                            setting_path = os.path.join(summarywriter_dir, 'train_setting.json')
                                            json.dump(args.__dict__, open(setting_path, 'w'), indent=4)
                                            model_path = os.path.join(args.experiment_dir, 'models', args.weight_name)
                                            trainer.save_weights(model_path = model_path)