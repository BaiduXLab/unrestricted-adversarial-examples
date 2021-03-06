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

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test, attack_over_test_data
from models.models import edge_resnet18, edge_resnet50, Contrastive_res50, resnet50_ori
from model_train_eval import AverageMeter, accuracy, save_checkpoint, get_model
from unrestricted_advex import eval_kit, attacks
from loss_func.loss_functions import FocalLoss
from loss_func.trades import trades_loss

import pdb

def load_data(param):
    # Data loading code
    data_path = '/home/yantao/datasets/bird_or_bicycle/0.0.4/'
    
    traindirs = [os.path.join(data_path, partition) for partition in ['extras']]
    # Use train as validation because it is IID with the test set
    valdir = os.path.join(data_path, 'train')
    testdir = os.path.join(data_path, 'test')

    # this normalization is NOT used, as the attack API requires
    # the images to be in [0, 1] range. So we prepend a BatchNorm
    # layer to the model instead of normalizing the images in the
    # data iter.
    _unused_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

    train_dataset = [datasets.ImageFolder(traindir, transforms.Compose([
                                            #transforms.RandomResizedCrop(224),
                                            transforms.RandomCrop(224, padding=16),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            _unused_normalize,
                                            ]))
                        for traindir in traindirs]
    if len(train_dataset) == 1:
        train_dataset = train_dataset[0]
    else:
        train_dataset = torch.utils.data.ConcatDataset(train_dataset)

    loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True, num_workers=param['workers'], pin_memory=True)

    loader_val = torch.utils.data.DataLoader(datasets.ImageFolder(valdir, transforms.Compose([  transforms.Resize(256),
                                                                                                transforms.CenterCrop(224),
                                                                                                transforms.ToTensor(),
                                                                                                _unused_normalize,
                                                                                                ])),
                                            batch_size=param['batch_size_eval'], shuffle=False,
                                            num_workers=param['workers_eval'], pin_memory=True)

    loader_test = torch.utils.data.DataLoader(datasets.ImageFolder(testdir, transforms.Compose([  transforms.Resize(256),
                                                                                                transforms.CenterCrop(224),
                                                                                                transforms.ToTensor(),
                                                                                                _unused_normalize,
                                                                                                ])),
                                             batch_size=param['batch_size_eval'], shuffle=False,
                                             num_workers=param['workers_eval'], pin_memory=True)
    
    return loader_train, loader_val, loader_test

def write_para_info(param, PGD_param, filepath = './para_info.txt'):
    with open(filepath, 'w') as file:
        file.write("param: " + '\n')
        for key, value in param.items():
            file.write(key + " : " + str(value) + '\n')
        file.write('\n')
        file.write("PGDparam: " + '\n')
        for key, value in PGD_param.items():
            file.write(key + " : " + str(value) + '\n')

def main():
    # Hyper-parameters
    is_evaluate_PGD = False
    is_evaluate_black = False
    param = {
        'batch_size': 8,
        'batch_size_eval' : 8,
        'num_epochs': 200,
        'delay': 0,
        'learning_rate': 1e-1,   # 1e-4
        'weight_decay': 5e-4,
        'w_l1' : 0,
        'momentum' : 0.9,
        'trades' : True,
        'trades_beta' : 1.,
        'net_type' : 'resnet50_ori',
    }
    PGD_param = {"epsilon" : 16. / 255,
                 "k" : 4,
                 "a" : 4. / 255,
                 "random_start" : True}

    param['workers'] = int(4 * (param['batch_size'] / 256))
    param['workers_eval'] = int(4 * (param['batch_size_eval'] / 256))

    # load bird or bicycle data
    loader_train, loader_val, loader_test = load_data(param)

    # load model
    if param['net_type'] == 'edge_resnet18':
        net = edge_resnet18()
    elif param['net_type'] == 'edge_resnet50':
        net = edge_resnet50()
    elif param['net_type'] == 'resnet50_fc':
        net = get_model()
    elif param['net_type'] == 'contrast_resnet50':
        net = Contrastive_res50()
    elif param['net_type'] == 'resnet50_ori':
        net = resnet50_ori()
    else:
        raise ValueError('Invalid net name.')
    
    if torch.cuda.is_available():
        print('CUDA enabled.')
        if param['net_type'] == 'resnet50_fc':
            net.cuda()
        else:
            net = torch.nn.DataParallel(net).cuda()

    pre_weight_path = None
    #pre_weight_path = "/home/yantao/workspace/resnet50_ori/Tue Feb  5 21:30:29 2019/adv_43.pth.tar"
    param['pre_weight_path'] = pre_weight_path

    if pre_weight_path is not None:
        if os.path.isfile(pre_weight_path):
            print("=> loading checkpoint '{}'".format(pre_weight_path))
            checkpoint = torch.load(pre_weight_path)
            net.load_state_dict(checkpoint['state_dict'])
            if param['net_type'] != 'resnet50_fc':
                param['net_type'] = checkpoint['arch']
        else:
            raise ValueError('wight path does not exist.')
    
    if is_evaluate_PGD:
        PGD_param_eval = {"epsilon" : 16. / 255,
                          "k" : 40,
                          "a" : 2. / 255,
                          "random_start" : True,
                         }
        adversary = LinfPGDAttack(epsilon=PGD_param_eval["epsilon"], k=PGD_param_eval["k"], a=PGD_param_eval["a"], random_start=PGD_param_eval["random_start"])
        acc = attack_over_test_data(net, adversary, loader_val)
        print(PGD_param_eval)
        print("eval accuracy: " , str(acc))
        acc = attack_over_test_data(net, adversary, loader_test)
        print("test accuracy: " , str(acc))
        if not is_evaluate_black:
            quit()

    if is_evaluate_black:
        def wrapped_model(x_np):
                x_np = x_np.transpose((0, 3, 1, 2))  # from NHWC to NCHW
                x_t = torch.from_numpy(x_np).cuda()
                net.eval()
                with torch.no_grad():
                    result = net(x_t).cpu().numpy()
                return result
            
        eval_prec = eval_kit.evaluate_bird_or_bicycle_model(wrapped_model, model_name=param['net_type']+'_adv_eval') #_on_trainingset
        print(eval_prec)
        quit()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=param['learning_rate'], momentum=param['momentum'],  weight_decay=param['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100, 150], gamma=0.2)

    # create log
    data_time_str = datetime.now().ctime()
    '''
    save_weights_dir = os.path.join('/home/yantao/workspace', param['net_type'] + '_trades', data_time_str)
    os.mkdir(save_weights_dir)

    write_para_info(param, PGD_param, filepath = os.path.join(save_weights_dir, 'para_info.txt'))
    '''

    loss_file = './loss_info_' + param['net_type'] + '_trades_' + data_time_str + '.csv'
    with open(loss_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["loss", "loss_eval"])

    # Train the model
    print("Start training")
    for epoch in range(param['num_epochs']):
        net.train()
        loss_mean = AverageMeter()
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))
        for _, (x, y) in enumerate(tqdm(loader_train)):
            data, target = x.cuda(), y.cuda()

            optimizer.zero_grad()

            # calculate robust loss
            loss = trades_loss(model=net,
                            x_natural=data,
                            y=target,
                            optimizer=optimizer,
                            step_size=PGD_param['a'],
                            epsilon=PGD_param['epsilon'],
                            perturb_steps=PGD_param['k'],
                            batch_size=param['batch_size'],
                            beta=param['trades_beta'])
            loss.backward()
            optimizer.step()
            loss_mean.update(loss.cpu().detach().numpy())

        lr_scheduler.step()

        # get evaluate loss
        loss_eval_mean = AverageMeter()
        loss_pgd_eval_mean = AverageMeter()

        for _, (x, y) in enumerate(tqdm(loader_val)):
            net.eval()
            x_var, y_var = to_var(x), to_var(y.long())

            output = net(x_var)
            loss_eval = criterion(output, y_var)

            adversary = LinfPGDAttack(epsilon=PGD_param["epsilon"], k=PGD_param["k"], a=PGD_param["a"], random_start=PGD_param["random_start"])
            y_pred = pred_batch(x, net)
            x_adv = adv_train(x, y_pred, net, criterion, adversary)
            x_adv_var = to_var(x_adv)
            y_pred_adv = net(x_adv_var)
            loss_pgd_eval = criterion(y_pred_adv, y_var)

            loss_pgd_eval_mean.update(loss_pgd_eval.cpu().detach().numpy())
            loss_eval_mean.update(loss_eval.cpu().detach().numpy())

        print("loss , loss_eval, loss_pgd_eval: ", loss_mean.avg, " , ", loss_eval_mean.avg, " , ", loss_pgd_eval_mean.avg)
        # record loss info
        with open(loss_file, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([loss_mean.avg, loss_eval_mean.avg, loss_pgd_eval_mean.avg])

        def wrapped_model(x_np):
            x_np = x_np.transpose((0, 3, 1, 2))  # from NHWC to NCHW
            x_t = torch.from_numpy(x_np).cuda()
            net.eval()
            with torch.no_grad():
                result = net(x_t).cpu().numpy()
            return result
        
        
        if (epoch + 1) % 200 == 0:
            eval_prec = eval_kit.evaluate_bird_or_bicycle_model(wrapped_model, model_name='adv_'+str(epoch)) #_on_trainingset
            print("epoch: ", epoch)
            print(eval_prec)
        '''
        # save model for each epoch
        save_dic = {
            'arch' : param['net_type'],
            'state_dict' : net.state_dict()
        }
        save_model(save_dic, filename=os.path.join(save_weights_dir, 'adv_' + str(epoch) + '.pth.tar'))
        '''

def save_model(save_dic, filename):
    torch.save(save_dic, filename)
        






if __name__ == '__main__':
    main()