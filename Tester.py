import os
import numpy as np
import torch
from tqdm import tqdm

from dataloaders.dataloader import build_test_dataloader
from model.lstmnet import LSTMNet
from model.criterion import build_criterion

from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_score

class Tester(object):

    def __init__(self, args):
        self.args = args
        kargs = {'num_workers': args.workers}
        self.test_loader = build_test_dataloader(args, **kargs)

        self.model = LSTMNet(args) # 定义网络
        self.criterion = build_criterion(args)
    
        if args.cuda:
           self.model = self.model.cuda()
           self.criterion = self.criterion.cuda()
    
    def load_state_dict(self, state_dict_path):
        self.model.load_state_dict(torch.load(state_dict_path))

    def eval(self):
        """测试所有测试集
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

        roc_auc = roc_auc_score(total_target, total_pred)
        pr_auc = average_precision_score(total_target, total_pred)
        print("ROC-AUC: %.4f, PR-AUC: %.4f." % (roc_auc, pr_auc))
        # rscore = recall_score(total_target , total_pred)
        # pscore = precision_score(total_target, total_pred)
        # print("ROC-AUC: %.4f, PR-AUC: %.4f, RSCORE: %.4f, PSCORE: %.4f." % (roc_auc, pr_auc, rscore, pscore))

        return roc_auc, pr_auc, test_loss
        # return roc_auc, pr_auc, rscore, pscore, test_loss