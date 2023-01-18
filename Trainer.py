import os
import numpy as np
import torch
from tqdm import tqdm

from dataloaders.dataloader import build_dataloader
from model.lstmnet import LSTMNet
from model.criterion import build_criterion

from utils import aucPerformance

class Trainer(object):

    def __init__(self, args):
        self.args = args
        kargs = {'num_workers': args.workers}
        self.train_loader, self.test_loader = build_dataloader(args, **kargs)
        self.model = LSTMNet(args) # 定义网络
        self.criterion = build_criterion(args)
        if args.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

        if args.cuda:
           self.model = self.model.cuda()
           self.criterion = self.criterion.cuda()

    def train(self, epoch : int):
        """训练一个epoch
        Args:
            epoch (int): 当前epoch
        """
        self.model.train()

        train_loss = 0.0
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            batch_data, label = sample['data'], sample['label']
            if self.args.cuda:
                batch_data, label = batch_data.cuda(), label.cuda()

            output = self.model(batch_data)
            loss = self.criterion(output, label.unsqueeze(1).float())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (i + 1)))
        self.scheduler.step()

        cur_epoch_loss = train_loss / (i + 1)
        return cur_epoch_loss

    def eval(self):
        """测试一个args.steps_per_epoch个batch的数据
        异常检测更加关注异常样本的检测情况，这里主要关注PR曲线，即精确度和召回率。
        """
        self.model.eval()

        test_loss = 0.0
        tbar = tqdm(self.test_loader, desc='\r')
        total_pred = np.array([])
        total_target = np.array([])
        for i, sample in enumerate(tbar):
            batch_data, label = sample['data'], sample['label']
            if self.args.cuda:
                batch_data, label = batch_data.cuda(), label.cuda()

            with torch.no_grad():
                output = self.model(batch_data)
            loss = self.criterion(output, label.unsqueeze(1).float())

            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            
            total_pred = np.append(total_pred, output.data.cpu().numpy())
            total_target = np.append(total_target, label.cpu().numpy())

        cur_epoch_loss = test_loss / (i + 1)
        roc_auc, pr_auc = aucPerformance(total_pred, total_target)
        return roc_auc, pr_auc, cur_epoch_loss

    def save_weights(self, filename = None):
        if filename == None:
            torch.save(self.model.state_dict(), os.path.join(self.args.experiment_dir, self.weight_name))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.args.experiment_dir, filename))