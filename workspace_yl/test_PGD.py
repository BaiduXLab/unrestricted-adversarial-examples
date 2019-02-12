import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
from datetime import datetime
from tqdm import tqdm
import csv
import numpy as np
import pickle
import random

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test, attack_over_test_data
from models.models import Contrastive_res50_fe, resnet50_ori
from model_train_eval import AverageMeter, accuracy, save_checkpoint, get_model
from unrestricted_advex import eval_kit, attacks
from loss_func.loss_functions import ContrastiveLoss
from model_adv_train_2 import load_data, write_para_info, save_model
from torch.utils.data import DataLoader, Dataset

import pdb


def main():
    # Hyper-parameters
    is_debug = True
    is_evaluate_PGD = False
    is_evaluate_black = False
    param = {
        'batch_size': 256,
        'batch_size_eval' : 8,
        'num_epochs': 200,
        'delay': 30,
        'learning_rate': 1e-1,   # 1e-4
        'weight_decay': 5e-4,
        'w_l1' : 0,
        'momentum' : 0.9,
        'focal_loss' : False,
        'focal_loss_gamma' : 1,
        'train_ori' : False,
        'adv_PGD' : True,
        'net_type' : 'resnet50_ori',
        
    }
    PGD_param = {"epsilon" : 16. / 255,
                 "k" : 8,
                 "a" : 2. / 255,
                 "random_start" : True}

    param['workers'] = int(4 * (param['batch_size'] / 256))
    param['workers_eval'] = int(4 * (param['batch_size_eval'] / 256))

    # load bird or bicycle data
    loader_train, loader_val, loader_test = load_data(param)

    #net = resnet50_ori(isPretrain=True)
    net = torchvision.models.resnet50(num_classes=2)

    net = torch.nn.DataParallel(net).cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=param['learning_rate'], momentum=param['momentum'],  weight_decay=param['weight_decay'])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.2)

    for epoch in range(param['num_epochs']):
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))
        net.train()
        loss_ori_mean = AverageMeter()
        for _, (x, y) in enumerate(tqdm(loader_train)):
            x_var, y_var = to_var(x), to_var(y.long())
            loss_ori = criterion(net(x_var), y_var)
            optimizer.zero_grad()
            loss_ori.backward()
            optimizer.step()

            loss_ori_mean.update(loss_ori.cpu().detach().numpy())

        lr_scheduler.step()

        loss_eval_mean = AverageMeter()
        for _, (x, y) in enumerate(tqdm(loader_val)):
            net.eval()
            x_var, y_var = to_var(x), to_var(y.long())

            output = net(x_var)
            loss_eval = criterion(output, y_var)
            
            loss_eval_mean.update(loss_eval.cpu().detach().numpy())

        print("loss_ori, loss_ori_eval: ", loss_ori_mean.avg, " , " , loss_eval_mean.avg)

        save_dic = {
            'state_dict' : net.state_dict()
        }
        torch.save(save_dic, 'clean_test/adv_' + str(epoch) + '.pth.tar')
    
    PGD_param_eval = {"epsilon" : 16. / 255,
                          "k" : 8,
                          "a" : 2. / 255,
                          "random_start" : True,
                         }
    adversary = LinfPGDAttack(epsilon=PGD_param_eval["epsilon"], k=PGD_param_eval["k"], a=PGD_param_eval["a"], random_start=PGD_param_eval["random_start"])
    acc = attack_over_test_data(net, adversary, loader_val)
    print(PGD_param_eval)
    print("eval accuracy: " , str(acc))
    acc = attack_over_test_data(net, adversary, loader_test)
    print("test accuracy: " , str(acc))

if __name__ == '__main__':
    main()
