from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve
import numpy as np

def get_metric(targets, preds):
    auc = roc_auc_score(targets, preds)
    acc = accuracy_score(targets, np.where(preds >= 0.5, 1, 0))
    precision=precision_score(targets, np.where(preds >= 0.5, 1, 0))
    recall=recall_score(targets, np.where(preds >= 0.5, 1, 0))
    f1=f1_score(targets, np.where(preds >= 0.5, 1, 0))
    
    return auc, acc ,precision,recall,f1