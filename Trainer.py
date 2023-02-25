import os
import numpy as np
import torch
from tqdm import tqdm
import time

from dataloaders.dataloader import build_train_dataloader, build_valid_dataloader
from model.net import Net
from model.criterion import build_criterion

from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

class Trainer(object):

    def __init__(self, args):
        self.args = args
        kargs = {'num_workers': args.workers}
        self.train_loader, self.valid_loader = build_train_dataloader(args, **kargs)
        self.test_loader = build_valid_dataloader(args, **kargs)
        self.train_num = len(self.train_loader.dataset)

        self.model = Net(args) # 定义网络
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
        Returns:
            cur_epoch_loss (float): 当前epoch的loss
        """
        self.model.train()

        train_loss = 0.0
        tbar = tqdm(self.train_loader)
        for i, sample in enumerate(tbar):
            batch_data, label, valid_lens = sample['data'], sample['label'], sample['valid_lens']
            # print("batch_data:", batch_data.shape, batch_data[0])
            # input()
            if self.args.cuda:
                batch_data, label, valid_lens = batch_data.cuda(), label.cuda(), valid_lens.cuda()

            output = self.model(batch_data, valid_lens)
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

    def eval(self, epoch):
        """测试一个args.steps_per_epoch个batch的数据
        异常检测更加关注异常样本的检测情况，这里主要关注PR曲线，即精确度和召回率。
        Args:
            epoch (int): 当前epoch
        Returns:
            roc_auc (float): 当前epoch的roc_auc
            pr_auc (float): 当前epoch的pr_auc
            cur_epoch_loss (float): 当前epoch的loss
        """
        self.model.eval()

        test_loss = 0.0
        tbar = tqdm(self.valid_loader, desc='\r')
        total_pred = np.array([])
        total_target = np.array([])
        total_uid = np.array([])
        for i, sample in enumerate(tbar):
            batch_data, label, valid_lens = sample['data'], sample['label'], sample['valid_lens']
            uid = sample['uid']
            if self.args.cuda:
                batch_data, label, valid_lens = batch_data.cuda(), label.cuda(), valid_lens.cuda()

            with torch.no_grad():
                output = self.model(batch_data, valid_lens)
            loss = self.criterion(output, label.unsqueeze(1).float())

            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            
            total_pred = np.append(total_pred, output.data.cpu().numpy())
            total_target = np.append(total_target, label.cpu().numpy())
            total_uid = np.append(total_uid, uid)

        sort_index = np.argsort(-total_pred)
        total_pred = total_pred[sort_index]
        total_target = total_target[sort_index]
        total_uid = total_uid[sort_index]

        cur_epoch_loss = test_loss / (i + 1)
        try:
            roc_auc = roc_auc_score(total_target, total_pred)
            pr_auc = average_precision_score(total_target, total_pred)
            precision, recall, thresholds = precision_recall_curve(total_target, total_pred)
            F1 = 2 * precision * recall / (precision + recall)
            idx = np.argmax(F1)
            best_thresholds = precision[idx]
            best_precision = precision[idx]
            best_recall = recall[idx]
            best_F1 = F1[idx]
        except:
            roc_auc = 0
            pr_auc = 0

        # rscore = recall_score(total_target , total_pred)
        # pscore = precision_score(total_target, total_pred)

        # precision, recall, thresholds = precision_recall_curve(total_target, total_pred)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(recall, precision)
        # ax.set_xlabel('Recall')
        # ax.set_ylabel('Precision')
        # ax.set_title('PR Curve')
        # figures_dir = os.path.join(self.args.experiment_dir, "figures")
        # plt.savefig(os.path.join(figures_dir, f'pr_curve-valid-{epoch}-{self.args.cur_time}.png'), bbox_inches='tight')

        return roc_auc, pr_auc, best_precision, best_recall, best_F1, cur_epoch_loss, total_target, total_pred, total_uid
    
    def test(self):
        """测试所有测试集
        Args:
            state_dict_path (str): 模型参数路径
        """
        
        test_loss = 0.0
        tbar = tqdm(self.test_loader, desc='\r')
        total_pred = np.array([])
        total_target = np.array([])
        total_uid = np.array([])

        epoch_num = 0
        for i, sample in enumerate(tbar):
            batch_data, label, valid_lens = sample['data'], sample['label'], sample['valid_lens']
            uid = sample['uid']
            if self.args.cuda:
                batch_data, label, valid_lens = batch_data.cuda(), label.cuda(), valid_lens.cuda()

            with torch.no_grad():
                output = self.model(batch_data, valid_lens)
            loss = self.criterion(output, label.unsqueeze(1).float())

            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            
            total_pred = np.append(total_pred, output.data.cpu().numpy())
            total_target = np.append(total_target, label.cpu().numpy())
            total_uid = np.append(total_uid, uid)
            epoch_num += 1

        sort_index = np.argsort(-total_pred)
        total_pred = total_pred[sort_index]
        total_target = total_target[sort_index]
        total_uid = total_uid[sort_index]

        try:
            roc_auc = roc_auc_score(total_target, total_pred)
            pr_auc = average_precision_score(total_target, total_pred)
            precision, recall, thresholds = precision_recall_curve(total_target, total_pred)
            F1 = 2 * precision * recall / (precision + recall)
            idx = np.argmax(F1)
            best_thresholds = precision[idx]
            best_precision = precision[idx]
            best_recall = recall[idx]
            best_F1 = F1[idx]
        except:
            roc_auc = 0
            pr_auc = 0

        # precision, recall, thresholds = precision_recall_curve(total_target, total_pred)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(recall, precision)
        # ax.set_xlabel('Recall')
        # ax.set_ylabel('Precision')
        # ax.set_title('PR Curve')
        # figures_dir = os.path.join(self.args.experiment_dir, "figures")
        # plt.savefig(os.path.join(figures_dir, f'pr_curve-test.png'), bbox_inches='tight')

        return roc_auc, pr_auc, best_precision, best_recall, best_F1,  test_loss / epoch_num, total_target, total_pred, total_uid
    
    def save_weights(self, model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.model.state_dict(), model_path)

    def load_weights(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
    
    def get_weights(self):
        return self.model.state_dict()