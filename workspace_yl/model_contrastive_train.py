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
from models.models import Contrastive_res50_fe
from model_train_eval import AverageMeter, accuracy, save_checkpoint, get_model
from unrestricted_advex import eval_kit, attacks
from loss_func.loss_functions import ContrastiveLoss
from model_adv_train_2 import load_data, write_para_info, save_model
from torch.utils.data import DataLoader, Dataset

import pdb

class ContrastNetworkDataset(Dataset):
    
    def __init__(self, imageFolderDataset, transform=None, should_invert=False):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageFolderDataset) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break
        
        return img0_tuple[0], img1_tuple[0] , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

def main():
    # Hyper-parameters
    is_evaluate_PGD = False
    is_evaluate_black = False
    param = {
        'batch_size': 8,
        'batch_size_eval' : 8,
        'num_epochs': 200,
        'delay': 0,
        'learning_rate': 1e-2,   # 1e-4
        'weight_decay': 5e-4,
        'w_l1' : 0,
        'momentum' : 0.9,
        'focal_loss' : False,
        'focal_loss_gamma' : 1,
        'train_ori' : False,
        'adv_PGD' : True,
        'adv_spatial_SPSA' : False,
        'net_type' : 'contrast_resnet50',
        
    }
    PGD_param = {"epsilon" : 16. / 255,
                 "k" : 8,
                 "a" : 2. / 255,
                 "random_start" : True}

    param['workers'] = int(4 * (param['batch_size'] / 256))
    param['workers_eval'] = int(4 * (param['batch_size_eval'] / 256))

    # load bird or bicycle data
    #loader_train, loader_val, loader_test = load_data(param)
    data_path = '/home/yantao/datasets/bird_or_bicycle/0.0.4/'
    
    traindir = os.path.join(data_path, 'extras')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize,
                                            ]))
    contrast_loader = ContrastNetworkDataset(imageFolderDataset=train_dataset)

    train_dataloader = DataLoader(contrast_loader,
                                    shuffle=True,
                                    num_workers=param['workers'],
                                    batch_size=param['batch_size'])

    net = Contrastive_res50_fe()

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

    criterion = ContrastiveLoss(margin=0.8).cuda()

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
        writer.writerow(["loss"])

    # Train the model
    print("Start training")
    for epoch in range(param['num_epochs']):
        net.train()
        loss_mean = AverageMeter()
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))
        for _, (x0, x1, y) in enumerate(tqdm(train_dataloader)):
            y = y.cuda()
            y0, y1 = net(x0, x1)
            loss = criterion(y0, y1, y)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_mean.update(loss.cpu().detach().numpy())
        lr_scheduler.step()
        print('loss: ', loss_mean.avg)
        with open(loss_file, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([loss_mean.avg])
        save_dic = {
            'arch' : param['net_type'],
            'state_dict' : net.state_dict()
        }
        save_model(save_dic, filename=os.path.join(save_weights_dir, 'contrast_fe_' + str(epoch) + '.pth.tar'))




if __name__ == '__main__':
    main()