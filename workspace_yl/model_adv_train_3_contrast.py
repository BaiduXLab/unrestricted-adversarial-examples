import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
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
    is_evaluate_PGD = False
    is_evaluate_black = False
    param = {
        'batch_size': 32,
        'batch_size_eval' : 8,
        'num_epochs': 200,
        'delay': 0,
        'learning_rate': 1e-1,   # 1e-4
        'weight_decay': 5e-4,
        'w_l1' : 0,
        'momentum' : 0.9,
        'adv_PGD' : True,
        'adv_spatial_SPSA' : False,
        'net_type' : 'contrast_resnet50',
        'contrast_beta' : 1.0,
        
    }
    PGD_param = {"epsilon" : 16. / 255,
                 "k" : 4,
                 "a" : 4. / 255,
                 "random_start" : True}

    param['workers'] = int(4 * (param['batch_size'] / 256))
    param['workers_eval'] = int(4 * (param['batch_size_eval'] / 256))

    # load bird or bicycle data
    loader_train, loader_val, loader_test = load_data(param)

    net = resnet50_ori(n_channels=3, num_classes=2, fe_branch=True, isPretrain=True)

    if torch.cuda.is_available():
        print('CUDA enabled.')
        net = torch.nn.DataParallel(net).cuda()

    pre_weight_path = None
    #pre_weight_path = "/home/yantao/workspace/adv_training_models/yunhan_focal_loss/undefended_pytorch_resnet_adv_120.pth.tar"
    if pre_weight_path is not None:
        if os.path.isfile(pre_weight_path):
            print("=> loading checkpoint '{}'".format(pre_weight_path))
            checkpoint = torch.load(pre_weight_path)
            net.load_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError('wight path does not exist.')

    optimizer = torch.optim.SGD(net.parameters(), lr=param['learning_rate'], momentum=param['momentum'],  weight_decay=param['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)

    # create log
    data_time_str = datetime.now().ctime()
    save_weights_dir = os.path.join('/home/yantao/workspace', param['net_type'], data_time_str)
    os.mkdir(save_weights_dir)

    write_para_info(param, PGD_param, filepath = os.path.join(save_weights_dir, 'para_info.txt'))

    loss_file = './loss_info_' + param['net_type'] + '_' + data_time_str + '.csv'
    with open(loss_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["loss", "loss_ori", "loss_contrast", "loss_eval_ori", "loss_eval_pgd"])

    # Train the model
    print("Start training")
    for epoch in range(param['num_epochs']):
        net.train()
        loss_mean = AverageMeter()
        loss_ori_mean = AverageMeter()
        loss_contrast_mean = AverageMeter()
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))
        for batch_idx, (x, y) in enumerate(tqdm(loader_train)):
            x_var, y_var = to_var(x), to_var(y.long())
            logits, fe = net(x_var)
            loss_ori = F.cross_entropy(logits, y_var)

            adversary = LinfPGDAttack(epsilon=PGD_param["epsilon"], k=PGD_param["k"], a=PGD_param["a"], random_start=PGD_param["random_start"])
            # use predicted label to prevent label leaking

            x_adv = adv_train(x, y, net, None, adversary, multi_out=True)
            x_adv_var = to_var(x_adv)
            logits_adv, fe_adv = net(x_adv_var)
            loss_contrast = F.mse_loss(fe_adv, fe)

            loss = loss_ori + param['contrast_beta'] * loss_contrast
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_mean.update(loss.cpu().detach().numpy())
            loss_ori_mean.update(loss_ori.cpu().detach().numpy())
            loss_contrast_mean.update(loss_contrast.cpu().detach().numpy())
            if batch_idx % 30 == 0:
                print(loss.cpu().detach().numpy())
                print(loss_ori.cpu().detach().numpy())
                print(loss_contrast.cpu().detach().numpy())
        lr_scheduler.step()

        # get evaluate loss
        loss_eval_mean = AverageMeter()
        loss_pgd_eval_mean = AverageMeter()
        for _, (x, y) in enumerate(tqdm(loader_val)):
            net.eval()
            criterion = nn.CrossEntropyLoss().cuda()
            pdb.set_trace()
            x_var, y_var = to_var(x), to_var(y.long())
            output, _ = net(x_var)
            loss_eval = criterion(output, y_var)

            if param['adv_PGD']:
                adversary = LinfPGDAttack(epsilon=PGD_param["epsilon"], k=PGD_param["k"], a=PGD_param["a"], random_start=PGD_param["random_start"])
                # y_pred = pred_batch(x, net)
                x_adv = adv_train(x, y, net, criterion, adversary, multi_out=True)
                x_adv_var = to_var(x_adv)
                y_pred_adv, _ = net(x_adv_var)
                loss_pgd_eval = criterion(y_pred_adv, y_var)

                loss_pgd_eval_mean.update(loss_pgd_eval.cpu().detach().numpy())
            
            loss_eval_mean.update(loss_eval.cpu().detach().numpy())

        print("loss , loss_ori, loss_contrast, loss_eval, loss_pgd_eval: ", loss_mean.avg, " , ", loss_ori_mean.avg, " , ", loss_contrast_mean.avg, " , ", loss_eval_mean.avg, " , ", loss_pgd_eval_mean.avg)
        with open(loss_file, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([loss_mean.avg, loss_ori_mean.avg, loss_contrast_mean.avg, loss_eval_mean.avg, loss_pgd_eval_mean.avg])

        save_dic = {
            'arch' : param['net_type'],
            'state_dict' : net.state_dict()
        }
        save_model(save_dic, filename=os.path.join(save_weights_dir, 'contrast_fe_' + str(epoch) + '.pth.tar'))




if __name__ == '__main__':
    main()