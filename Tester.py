import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

from dataloaders.dataloader import build_valid_dataloader,  build_test_dataloader
from model.net import Net
from model.criterion import build_criterion

class Tester(object):

    def __init__(self, args):
        self.args = args
        kargs = {'num_workers': args.workers}

        if args.test_set == "valid":
            self.test_loader = build_valid_dataloader(args, **kargs)
        else:
            self.test_loader = build_test_dataloader(args, **kargs)

        self.model = Net(args) # 定义网络
        self.criterion = build_criterion(args)
    
        if args.cuda:
           self.model = self.model.cuda()
           self.criterion = self.criterion.cuda()
    
    def eval(self, state_dict_path : str):
        """测试所有测试集
        Args:
            state_dict_path (str): 模型参数路径
        """
        self.model.load_state_dict(torch.load(state_dict_path))
        
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