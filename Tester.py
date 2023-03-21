import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

from dataloaders.dataloader import build_valid_dataloader,  build_test_dataloader, build_train_dataloader
from model.net import Net
from model.criterion import build_criterion

class Tester(object):

    def __init__(self, args):
        self.args = args
        kargs = {'num_workers': args.workers}
        
        if args.test_set == "train":
            self.test_loader = build_train_dataloader(args, **kargs)
        elif args.test_set == "valid":
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
                if self.args.criterion != "Contrastive":
                    output = self.model(batch_data, valid_lens)
                    loss = self.criterion(output, label.unsqueeze(1).float())
                else:
                    X, output = self.model(batch_data, valid_lens)
                    bce_loss, con_loss = self.criterion(X, output, label.unsqueeze(1).float())
                    loss = bce_loss + con_loss

            test_loss += loss.item()
            if self.args.criterion != "Contrastive":
                tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            else:
                tbar.set_description('Test loss: %.3f, %.3f, %.3f' % (loss, bce_loss, con_loss))
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

        normal_sample = np.sum(total_target == 0)
        abnormal_sample = np.sum(total_target == 1)
        print("normal data vs. abnormal data: ", normal_sample, abnormal_sample, "ratio: ", abnormal_sample / (normal_sample + abnormal_sample))
        # 20: normal data vs. abnormal data:  403142 6428 ratio:  0.015694508875161755
        # 50: normal data vs. abnormal data:  440255 10030 ratio: 0.022274781527254962
        # 100: normal data vs. abnormal data:  200878 7284 ratio: 0.03499197740221558   76.3900 = 0.00036697379925250526
        # 200: normal data vs. abnormal data:  89278 4790 ratio:  0.05092061062210316   34.7053 = 0.00036893842752051706
        # 300: normal data vs. abnormal data:  32102 2323 ratio:  0.0674800290486565    13.0781 = 0.0003799012345679012
        # all: normal data vs. abnormal data:  1165655 30855 ratio:  0.02578749864188348 444.1302 = 0.0003711880385454363

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