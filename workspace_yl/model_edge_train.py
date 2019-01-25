import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test, attack_over_test_data
from models.models import edge_resnet18
from model_train_eval import AverageMeter, accuracy, save_checkpoint
from unrestricted_advex import eval_kit, attacks
from loss_func.loss_functions import FocalLoss
import time
import os
from datetime import datetime
from tqdm import tqdm
import csv
import numpy as np
import pickle

import pdb

def load_data(param):
    # Data loading code
    data_path = '/home/yantao/datasets/bird_or_bicycle/0.0.4/'
    
    traindirs = [os.path.join(data_path, partition) for partition in ['extras']]
    # Use train as validation because it is IID with the test set
    valdir = os.path.join(data_path, 'train')

    # this normalization is NOT used, as the attack API requires
    # the images to be in [0, 1] range. So we prepend a BatchNorm
    # layer to the model instead of normalizing the images in the
    # data iter.
    _unused_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

    train_dataset = [datasets.ImageFolder(traindir, transforms.Compose([
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),# _unused_normalize,
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
                                                                                                # _unused_normalize,
                                                                                                ])),
                                            batch_size=param['batch_size'], shuffle=False,
                                            num_workers=param['workers'], pin_memory=True)
    return loader_train, loader_val

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
        'batch_size': 4,
        'num_epochs': 200,
        'delay': 0,
        'learning_rate': 1e-2,   #1e-4
        'weight_decay': 5e-4,
        'momentum' : 0.9,
        'focal_loss' : False,
        'adv_PGD' : True,
        'adv_spatial_SPSA' : False,
        'net_type' : 'edge_resnet18',
        
    }
    PGD_param = {"epsilon" : 32. / 255,
                 "k" : 4,
                 "a" : 1. / 255,
                 "random_start" : True}

    param['workers'] = int(4 * (param['batch_size'] / 256))

    # load bird or bicycle data
    loader_train, loader_val = load_data(param)

    # load model
    if param['net_type'] == 'edge_resnet18':
        net = edge_resnet18()

    if torch.cuda.is_available():
        print('CUDA enabled.')
        #net.cuda()
        net = torch.nn.DataParallel(net).cuda()

    
    pre_weight_path = "/home/yantao/workspace/edge_resnet18/Thu Jan 24 03:41:28 2019/undefended_pytorch_resnet_adv_100.pth.tar"
    if os.path.isfile(pre_weight_path):
        print("=> loading checkpoint '{}'".format(pre_weight_path))
        checkpoint = torch.load(pre_weight_path)
        net.load_state_dict(checkpoint['state_dict'])
        param['net_type'] = checkpoint['arch']
    
    
    if is_evaluate_black:
        def wrapped_model(x_np):
                x_np = x_np.transpose((0, 3, 1, 2))  # from NHWC to NCHW
                x_t = torch.from_numpy(x_np).cuda()
                net.eval()
                with torch.no_grad():
                    result = net(x_t).cpu().numpy()
                return result
            
        eval_prec = eval_kit.evaluate_bird_or_bicycle_model(wrapped_model, model_name='edge_resnet18_adv_eval') #_on_trainingset

        print(eval_prec)
        quit()

    if is_evaluate_PGD:
        PGD_param_eval = {"epsilon" : 16. / 255,
                          "k" : 8,
                          "a" : 2. / 255,
                          "random_start" : True,
                         }
        adversary = LinfPGDAttack(epsilon=PGD_param_eval["epsilon"], k=PGD_param_eval["k"], a=PGD_param_eval["a"], random_start=PGD_param_eval["random_start"])
        acc = attack_over_test_data(net, adversary, loader_val)
        print(PGD_param_eval)
        print("accuracy: " , str(acc))
        quit()
    
    if param['focal_loss']:
        criterion = FocalLoss(gamma=2).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    #optimizer = torch.optim.SGD(net.parameters(), lr=param['learning_rate'], momentum=param['momentum'],  weight_decay=param['weight_decay'])
    optimizer = torch.optim.Adam(net.parameters(), lr=param['learning_rate'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)

    # create log
    data_time_str = datetime.now().ctime()
    save_weights_dir = os.path.join('/home/yantao/workspace', param['net_type'], data_time_str)
    os.mkdir(save_weights_dir)

    write_para_info(param, PGD_param, filepath = os.path.join(save_weights_dir, 'para_info.txt'))

    loss_file = './loss_info_' + data_time_str + '.csv'
    with open(loss_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["loss_ori", "loss_pgd", "loss_spatial_SPSA_mean", "loss_ori_eval", "loss_pgd_eval_mean"])

    # Train the model
    print("Start training")
    for epoch in range(param['num_epochs']):
        net.train()
        loss_ori_mean = AverageMeter()
        loss_pgd_mean = AverageMeter()
        loss_spatial_SPSA_mean = AverageMeter()
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))
        for _, (x, y) in enumerate(tqdm(loader_train)):
            x_var, y_var = to_var(x), to_var(y.long())
            loss_ori = criterion(net(x_var), y_var)
            
            optimizer.zero_grad()
            loss_ori.backward()
            optimizer.step()
            
            # adversarial training
            if epoch + 1 > param['delay'] and param['adv_PGD']:
                adversary = LinfPGDAttack(epsilon=PGD_param["epsilon"], k=PGD_param["k"], a=PGD_param["a"], random_start=PGD_param["random_start"])
                # use predicted label to prevent label leaking
                y_pred = pred_batch(x, net)
                x_adv = adv_train(x, y_pred, net, criterion, adversary)
                x_adv_var = to_var(x_adv)
                loss_pgd = criterion(net(x_adv_var), y_var)

                loss = loss_pgd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_pgd_mean.update(loss_pgd.cpu().detach().numpy())

            if epoch + 1 > param['delay'] and param['adv_spatial_SPSA']:

                def wrapped_model(x_np):
                    x_np = x_np.transpose((0, 3, 1, 2))  # from NHWC to NCHW
                    x_t = torch.from_numpy(x_np).cuda()
                    net.eval()
                    with torch.no_grad():
                        result = net(x_t).cpu().numpy()
                    return result

                adversary = attacks.SpsaWithRandomSpatialAttack(wrapped_model, 
                                                                image_shape_hwc=(224, 224, 3), 
                                                                spatial_limits=[18, 18, 30], 
                                                                black_border_size=20, 
                                                                epsilon=(16. / 255), 
                                                                num_steps=32,
                                                               )
                
                x_adv = adversary(wrapped_model, x.numpy().transpose((0, 2, 3, 1)), y.numpy())
                x_adv = torch.from_numpy(x_adv.transpose((0, 3, 1, 2)))
                x_adv_var = to_var(x_adv)
                loss_spatial_SPSA = criterion(net(x_adv_var), y_var)

                loss = loss_spatial_SPSA
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_spatial_SPSA_mean.update(loss_spatial_SPSA.cpu().detach().numpy())
            


            loss_ori_mean.update(loss_ori.cpu().detach().numpy())

        lr_scheduler.step()

        # get evaluate loss
        loss_eval_mean = AverageMeter()
        loss_pgd_eval_mean = AverageMeter()
        for _, (x, y) in enumerate(tqdm(loader_val)):
            net.eval()
            x_var, y_var = to_var(x), to_var(y.long())

            output = net(x_var)
            loss_eval = criterion(output, y_var)

            if epoch % 1 ==0 and param['adv_PGD']:
                adversary = LinfPGDAttack(epsilon=PGD_param["epsilon"], k=PGD_param["k"], a=PGD_param["a"], random_start=PGD_param["random_start"])
                y_pred = pred_batch(x, net)
                x_adv = adv_train(x, y_pred, net, criterion, adversary)
                x_adv_var = to_var(x_adv)
                y_pred_adv = net(x_adv_var)
                loss_pgd_eval = criterion(y_pred_adv, y_var)

                loss_pgd_eval_mean.update(loss_pgd_eval.cpu().detach().numpy())
            
            loss_eval_mean.update(loss_eval.cpu().detach().numpy())

        print("loss_ori , loss_pgd , loss_spatial_SPSA_mean , loss_ori_eval, loss_pgd_eval: ", loss_ori_mean.avg, " , ", loss_pgd_mean.avg, " , ", loss_spatial_SPSA_mean.avg, " , " , loss_eval_mean.avg, " , ", loss_pgd_eval_mean.avg)
        # record loss info
        with open(loss_file, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([loss_ori_mean.avg, loss_pgd_mean.avg, loss_spatial_SPSA_mean.avg, loss_eval_mean.avg, loss_pgd_eval_mean.avg])

        def wrapped_model(x_np):
            x_np = x_np.transpose((0, 3, 1, 2))  # from NHWC to NCHW
            x_t = torch.from_numpy(x_np).cuda()
            net.eval()
            with torch.no_grad():
                result = net(x_t).cpu().numpy()
            return result
        
        
        if (epoch + 1) % 50 == 0:
            eval_prec = eval_kit.evaluate_bird_or_bicycle_model(wrapped_model, model_name='undefended_pytorch_resnet_adv_'+str(epoch)) #_on_trainingset
            print("epoch: ", epoch)
            print(eval_prec)

        # save model for each epoch
        save_dic = {
            'arch' : param['net_type'],
            'state_dict' : net.state_dict()
        }
        save_model(save_dic, filename=os.path.join(save_weights_dir, 'undefended_pytorch_resnet_adv_' + str(epoch) + '.pth.tar'))
        

def save_model(save_dic, filename):
    torch.save(save_dic, filename)
        






if __name__ == '__main__':
    main()