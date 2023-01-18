import os
import torch
from Tester import Tester
from argparser import args


if __name__ == '__main__':

    torch.manual_seed(args.random_seed)

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    tester = Tester(args)

    model_path = os.path.join(args.experiment_dir, 'models', 'model-2023-01-18-12-12-23.pkl')
    tester.load_state_dict(model_path)

    roc_auc, pr_auc, test_loss = tester.eval()
    print("AUC-ROC: %.4f, AUC-PR: %.4f, test_loss: %.4f" % (roc_auc, pr_auc, test_loss))

    # roc_auc, pr_auc, rscore, pscore, test_loss = tester.eval()
    # print("AUC-ROC: %.4f, AUC-PR: %.4f, rscore: %.4f, pscore: %.4f, test_loss: %.4f" % (roc_auc, pr_auc, rscore, pscore, test_loss))