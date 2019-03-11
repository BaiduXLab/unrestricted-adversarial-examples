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
import copy

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack, CommonCorruptionsAttack, LinfPGDAttack_v2
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test, attack_over_test_data
from models.models import Contrastive_res50_fe, resnet50_ori, resnet18_ori, resnet101_ori
from model_train_eval import AverageMeter, accuracy, save_checkpoint, get_model
from unrestricted_advex import eval_kit, attacks
from loss_func.loss_functions import ContrastiveLoss
from model_adv_train_2 import load_data, write_para_info, save_model
from torch.utils.data import DataLoader, Dataset
from loss_func.loss_functions import FocalLoss

import torchvision.models as torchmodels

import pdb


def main():
    # Hyper-parameters
    is_debug = True
    is_evaluate_PGD = False
    is_evaluate_black = False
    param = {
        'batch_size': 256,
        'batch_size_eval' : 8,
        'num_epochs': 100,
        'delay': 0,
        'learning_rate': 1e-2,
        'weight_decay': 5e-4, #5e-4,
        'w_l1' : 0.0,
        'focal_loss' : False,
        'focal_loss_gamma' : 1.5,
        'momentum' : 0.9,
        'adv_PGD' : True,
        'net_type' : 'contrast_resnet50',
        'contrast_beta' : 1e2,
        'resnet_pretrain' : False,
        
    }
    PGD_param = {"epsilon" : 4. / 255,
                 "k" : 4,
                 "a" : 2. / 255,
                 "random_start" : True}

    # .eval() has changed!

    param['workers'] = 4
    param['workers_eval'] = 4

    # load bird or bicycle data
    loader_train, loader_val, loader_test = load_data(param)
    if param['net_type'] == 'contrast_resnet50':
        net = resnet50_ori(n_channels=3, num_classes=2, fe_branch=True, isPretrain=param['resnet_pretrain'])
        #net = torchmodels.resnet50(pretrained=False, num_classes=2)
    elif param['net_type'] == 'contrast_resnet101':
        net = resnet101_ori(n_channels=3, num_classes=2, fe_branch=True, isPretrain=param['resnet_pretrain'])
    else:
        raise ValueError('Invalid net type.')

    if torch.cuda.is_available():
        print('CUDA enabled.')
        net = torch.nn.DataParallel(net).cuda()

    pre_weight_path = None
    #pre_weight_path = "/home/yantao/workspace/contrast_resnet50/base_clean/contrast_fe_30.pth.tar"
    param['pre_weight_path'] = pre_weight_path
    if pre_weight_path is not None:
        if os.path.isfile(pre_weight_path):
            print("=> loading checkpoint '{}'".format(pre_weight_path))
            checkpoint = torch.load(pre_weight_path)
            net.load_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError('wight path does not exist.')

    if is_evaluate_PGD:
        PGD_param_eval = {"epsilon" : 16. / 255,
                          "k" : 20,
                          "a" : 2. / 255,
                          "random_start" : True,
                         }
        adversary = LinfPGDAttack(epsilon=PGD_param_eval["epsilon"], k=PGD_param_eval["k"], a=PGD_param_eval["a"], random_start=PGD_param_eval["random_start"])
        net.eval()
        acc = attack_over_test_data(net, adversary, loader_val, multi_out=True)
        print(PGD_param_eval)
        print("eval accuracy: " , str(acc))
        acc = attack_over_test_data(net, adversary, loader_train, multi_out=True)
        print("test accuracy: " , str(acc))
        if not is_evaluate_black:
            quit()

    if is_evaluate_black:
        def wrapped_model(x_np):
                x_np = x_np.transpose((0, 3, 1, 2))  # from NHWC to NCHW
                x_t = torch.from_numpy(x_np).cuda()
                net.eval()
                with torch.no_grad():
                    y_pred, _ = net(x_t)
                    result = y_pred.cpu().numpy()
                return result
            
        eval_prec = eval_kit.evaluate_bird_or_bicycle_model(wrapped_model, model_name=param['net_type']+'_adv_eval') #_on_trainingset
        print(eval_prec)
        quit()


    optimizer = torch.optim.SGD(net.parameters(), lr=param['learning_rate'], momentum=param['momentum'],  weight_decay=param['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
    if param['focal_loss']:
        criterion = FocalLoss(gamma=param['focal_loss_gamma']).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    # create log
    data_time_str = datetime.now().ctime()
    if not is_debug:
        save_weights_dir = os.path.join('/home/yantao/workspace', param['net_type'], data_time_str)
        os.mkdir(save_weights_dir)
        write_para_info(param, PGD_param, filepath = os.path.join(save_weights_dir, 'para_info.txt'))

    loss_file = './loss_info_' + param['net_type'] + '_' + data_time_str + '.csv'
    with open(loss_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["loss", "loss_ori", "loss_adv", "loss_contrast", "loss_eval_ori", "loss_eval_pgd", "acc_ori" , "acc_pgd", "acc_ori_eval", "acc_pgd_eval"])
    
    adversary = LinfPGDAttack_v2(epsilon=PGD_param["epsilon"], k=PGD_param["k"], a=PGD_param["a"], random_start=PGD_param["random_start"], multi_out=True)
    # Train the model
    print("Start training")
    for epoch in range(param['num_epochs']):
        total_correct_ori = 0
        total_correct_pgd = 0
        total_samples = len(loader_train.dataset)
        loss_mean = AverageMeter()
        loss_ori_mean = AverageMeter()
        loss_adv_mean = AverageMeter()
        loss_contrast_mean = AverageMeter()

        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))
        for batch_idx, (x, y) in enumerate(tqdm(loader_train)):
            #net.eval()
            net.train()
            x_var = x.cuda()
            y_var = y.cuda()

            # net.state_dict()
            logits, fe = net(x_var)
            #logits = net(x_var)
            loss_ori = criterion(logits, y_var)
            logits_argmax = np.argmax(logits.data.cpu().numpy(), axis=1)
            total_correct_ori += (logits_argmax == y.numpy()).sum()

            
            # use predicted label to prevent label leaking
            # y_pred = pred_batch(x, net, multi_out=True)
            # y_pred = torch.from_numpy(logits_argmax)
            x_adv = adversary(x, y, net, criterion, 'eval')
            x_adv_var = x_adv.cuda()
            logits_adv, fe_adv = net(x_adv_var) 
            #logits_adv = net(x_adv_var) 
            loss_adv = criterion(logits_adv, y_var)
            logits_adv_argmax = np.argmax(logits_adv.data.cpu().numpy(), axis=1)
            total_correct_pgd += (logits_adv_argmax == y.numpy()).sum()
            #loss_contrast = F.mse_loss(fe_adv, fe)
            loss_contrast = F.l1_loss(fe_adv, fe, reduction='mean')

            #loss = loss_ori + loss_adv + param['contrast_beta'] * loss_contrast
            '''
            loss_adv = torch.tensor(0).cuda()
            loss_contrast = torch.tensor(0).cuda()
            '''
            
            if batch_idx < param['delay']:
                loss = loss_ori
            else:
                #loss = loss_ori + loss_adv + param['contrast_beta'] * loss_contrast
                loss = loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_mean.update(loss.cpu().detach().numpy())
            loss_ori_mean.update(loss_ori.cpu().detach().numpy())
            loss_adv_mean.update(loss_adv.cpu().detach().numpy())
            loss_contrast_mean.update(loss_contrast.cpu().detach().numpy())
            
            '''
            if batch_idx % 1 == 0:
                print(loss.cpu().detach().numpy())
                print(loss_ori.cpu().detach().numpy())
                print(loss_adv.cpu().detach().numpy())
                print(loss_contrast.cpu().detach().numpy())
            '''
        lr_scheduler.step()
        #print(loss_ori_mean.avg, loss_adv_mean.avg)

        acc_ori = total_correct_ori / total_samples
        acc_pgd = total_correct_pgd / total_samples
        # get evaluate loss and acc
        
        total_correct_ori_eval = 0
        total_correct_pgd_eval = 0
        total_samples_eval = len(loader_val.dataset)
        loss_eval_mean = AverageMeter()
        loss_pgd_eval_mean = AverageMeter()

        
        for _, (x, y) in enumerate(tqdm(loader_val)):
            net.eval()
            x_var, y_var = to_var(x), to_var(y.long())
            output, _ = net(x_var)
            #output = net(x_var)
            loss_eval = criterion(output, y_var)
            output_argmax = np.argmax(output.data.cpu().numpy(), axis=1)
            total_correct_ori_eval += (output_argmax == y.numpy()).sum()

            if param['adv_PGD']:
                x_adv = adversary(x, y, net, criterion, 'eval')
                x_adv_var = to_var(x_adv)
                y_pred_adv, _ = net(x_adv_var)
                #y_pred_adv = net(x_adv_var)
                loss_pgd_eval = criterion(y_pred_adv, y_var)
                output_argmax = np.argmax(y_pred_adv.data.cpu().numpy(), axis=1)
                total_correct_pgd_eval += (output_argmax == y.numpy()).sum()

                loss_pgd_eval_mean.update(loss_pgd_eval.cpu().detach().numpy())
            
            loss_eval_mean.update(loss_eval.cpu().detach().numpy())

            acc_ori_eval = total_correct_ori_eval / total_samples_eval
            acc_pgd_eval = total_correct_pgd_eval / total_samples_eval
        
        print("loss , loss_ori, loss_adv, loss_contrast, loss_eval, loss_pgd_eval: ", loss_mean.avg, " , ", loss_ori_mean.avg, " , ", loss_adv_mean.avg, " , ", loss_contrast_mean.avg, " , ", loss_eval_mean.avg, " , ", loss_pgd_eval_mean.avg)
        print("acc_ori , acc_pgd, acc_ori_eval, acc_pgd_eval: ", acc_ori, " , ", acc_pgd, " , ", acc_ori_eval, " , ", acc_pgd_eval)

        
        with open(loss_file, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([loss_mean.avg, loss_ori_mean.avg, loss_adv_mean.avg, loss_contrast_mean.avg, loss_eval_mean.avg, loss_pgd_eval_mean.avg, acc_ori, acc_pgd, acc_ori_eval, acc_pgd_eval])
        if not is_debug:
            save_dic = {
                'arch' : param['net_type'],
                'state_dict' : net.state_dict()
            }
            save_model(save_dic, filename=os.path.join(save_weights_dir, 'contrast_fe_' + str(epoch) + '.pth.tar'))
        



if __name__ == '__main__':
    main()