import os
import torch
from torch.utils.tensorboard import SummaryWriter

from Trainer import Trainer
from argparser import args
import json


if __name__ == '__main__':

    torch.manual_seed(args.random_seed)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    seting_dir = os.path.join(args.experiment_dir, 'jsons')
    if not os.path.exists(seting_dir):
        os.makedirs(seting_dir)
    setting_path = os.path.join(seting_dir, f'train_setting-{args.cur_time}.json')
    
    json.dump(args.__dict__, open(setting_path, 'w'), indent=4)

    trainer = Trainer(args)
    runs_dir = os.path.join(args.experiment_dir, 'runs')
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    writer = SummaryWriter(os.path.join(runs_dir, f'train-{args.cur_time}'))
    
    for cur_epoch in range(0, trainer.args.epochs):
        cur_train_loss = trainer.train(cur_epoch)
        writer.add_scalar("train_loss", cur_train_loss, cur_epoch)
        
        if (cur_epoch + 1) % 5 == 0:
            cur_roc, cur_pr, cur_test_loss = trainer.eval(cur_epoch)
            writer.add_scalar("test_roc", cur_roc, cur_epoch)
            writer.add_scalar("test_pr", cur_pr, cur_epoch)
            writer.add_scalar("test_loss", cur_test_loss, cur_epoch)

            # cur_roc, cur_pr, cur_rscore, cur_pscore, cur_test_loss = trainer.eval()
            # writer.add_scalar("test_roc", cur_roc, cur_epoch)
            # writer.add_scalar("test_pr", cur_pr, cur_epoch)
            # writer.add_scalar("test_loss", cur_test_loss, cur_epoch)
            # writer.add_scalar("test_rscore", cur_rscore, cur_epoch)
            # writer.add_scalar("test_pscore", cur_pscore, cur_epoch)
    
    trainer.save_weights()

    writer.flush()
    writer.close()