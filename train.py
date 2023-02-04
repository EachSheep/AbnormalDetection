import os
import torch
from torch.utils.tensorboard import SummaryWriter
import json
import time

from Trainer import Trainer
from argparser import args


if __name__ == '__main__':

    torch.manual_seed(args.random_seed)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    seting_dir = os.path.join(args.experiment_dir, 'jsons')
    if not os.path.exists(seting_dir):
        os.makedirs(seting_dir)
    setting_path = os.path.join(seting_dir, f'train_setting.json')

    trainer = Trainer(args)
    runs_dir = os.path.join(args.experiment_dir, 'runs')
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)

    summarywriter_dir = os.path.join(runs_dir, f'train')
    if os.path.exists(summarywriter_dir):
        for root, dirs, files in os.walk(summarywriter_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(summarywriter_dir)
    writer = SummaryWriter(summarywriter_dir)
    
    begin_time = time.time()
    for cur_epoch in range(0, trainer.args.epochs):
        cur_train_loss = trainer.train(cur_epoch)
        writer.add_scalars("loss", {"train" : cur_train_loss}, cur_epoch)
        
        if (cur_epoch + 1) % 5 == 0:
            cur_roc, cur_pr, cur_test_loss, cur_label, cur_predict  = trainer.eval(cur_epoch)
            writer.add_scalars("loss", {"test" : cur_train_loss}, cur_epoch)
            writer.add_scalar("eval/roc", cur_roc, cur_epoch)
            writer.add_scalar("eval/pr", cur_pr, cur_epoch)
            writer.add_pr_curve(f"pr_curve/pr_curve-{cur_epoch}", cur_label, cur_predict, global_step=cur_epoch)

    end_time = time.time()
    args.train_time_per_epoch = "{:.3f}".format((end_time - begin_time) / args.epochs)
    json.dump(args.__dict__, open(setting_path, 'w'), indent=4)
    
    trainer.save_weights()

    writer.flush()
    writer.close()