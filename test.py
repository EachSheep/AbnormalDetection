import os
import torch
from Tester import Tester
from argparser import args


if __name__ == '__main__':

    torch.manual_seed(args.random_seed)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    tester = Tester(args)

    model_dir = os.path.join(args.experiment_dir, 'models')
    model_path = os.path.join(model_dir, 'model-2023-01-20-21-57-52.pkl')
    roc_auc, pr_auc, test_loss = tester.eval(state_dict_path = model_path)
    print("AUC-ROC: %.4f, AUC-PR: %.4f, test_loss: %.4f" % (roc_auc, pr_auc, test_loss))

    # roc_auc, pr_auc, rscore, pscore, test_loss = tester.eval()
    # print("AUC-ROC: %.4f, AUC-PR: %.4f, rscore: %.4f, pscore: %.4f, test_loss: %.4f" % (roc_auc, pr_auc, rscore, pscore, test_loss))