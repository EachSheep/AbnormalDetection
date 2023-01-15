import os
import numpy as np
import torch
from tqdm import tqdm

from dataloaders.dataloader import build_dataloader
from modeling.lstmnet import LSTMNet
from modeling.layers import build_criterion

from utils import aucPerformance

class Trainer(object):

    def __init__(self, args):
        self.args = args
        kargs = {'num_workers': args.workers}
        self.train_loader, self.test_loader = build_dataloader(args, **kargs)
        self.model = LSTMNet(args) # 定义网络
        self.criterion = build_criterion(args.criterion)
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
            single_data, label = sample['data'], sample['label']
            if self.args.cuda:
                single_data, label = single_data.cuda(), label.cuda()

            output = self.model(single_data)
            loss = self.criterion(output, label.unsqueeze(1).float())
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Epoch:%d, Train loss: %.3f' % (epoch, train_loss / (i + 1)))
        self.scheduler.step()

    def eval(self):
        self.model.eval()

        test_loss = 0.0
        tbar = tqdm(self.test_loader, desc='\r')
        total_pred = np.array([])
        total_target = np.array([])
        for i, sample in enumerate(tbar):
            single_data, label = sample['data'], sample['label']
            if self.args.cuda:
                single_data, label = single_data.cuda(), label.cuda()

            with torch.no_grad():
                output = self.model(single_data.float())
            loss = self.criterion(output, label.unsqueeze(1).float())

            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            
            total_pred = np.append(total_pred, output.data.cpu().numpy())
            total_target = np.append(total_target, label.cpu().numpy())
        roc, ap = aucPerformance(total_pred, total_target)
        return roc, ap

    def save_weights(self, filename = None):
        if filename == None:
            torch.save(self.model.state_dict(), os.path.join(self.args.experiment_dir, self.weight_name))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.args.experiment_dir, filename))