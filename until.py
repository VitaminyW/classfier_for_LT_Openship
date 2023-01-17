import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F
import importlib
import pdb

class Logger:
    def __init__(self):
        self.log_path = open('logs.txt', 'w+')
        print('logger initlized!')
    
    def __call__(self,content):
        print(content, file=self.log_path, flush=True)
    
    def __del__(self):
        print('Logger done and close!')
        self.log_path.close()


def shot_acc(preds, labels, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False,num_class = 5):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())
    class_acc = []
    for i in range(num_class):
        class_acc.append(class_correct[i] / test_class_count[i])
    return class_acc


def F_measure(preds, labels):
    return f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')


def mic_acc_cal(preds, labels):
    acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1


def class_count(data):
    labels = np.array(data.dataset.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num
