#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.metrics import average_precision_score, roc_auc_score

def aucPerformance(mse, labels, prt=True):
    roc_auc = roc_auc_score(labels, mse)
    pr_auc = average_precision_score(labels, mse)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, pr_auc))
    return roc_auc, pr_auc