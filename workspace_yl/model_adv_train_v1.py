"""
Adversarially train self model, pytorch
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
from model_train_eval import get_model
from model_train_eval import AverageMeter, accuracy, save_checkpoint
from unrestricted_advex import eval_kit, attacks
import time
import os
from datetime import datetime
from tqdm import tqdm
import csv
import numpy as np
import pickle

import pdb


def validate_epoch(val_loader, model, criterion, gpu=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if gpu is not None:
                input = input.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            '''
            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))
            '''

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def main():
    # Hyper-parameters
    param = {
        'batch_size': 8,
        'test_batch_size': 100,
        'num_epochs': 200,
        'delay': 0,
        'learning_rate': 1e-6,   #1e-4
        'weight_decay': 5e-4,
        'momentum' : 0.9,
        'adv_PGD' : True,
        'adv_CC' : False,
    }
    PGD_param = {"epsilon" : 32. / 255,
                 "k" : 8,
                 "a" : 2. / 255,
                 "random_start" : True}

    param['workers'] = int(4 * (param['batch_size'] / 256))


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


    # Setup the model
    net = get_model()
    criterion = nn.CrossEntropyLoss().cuda()

    pre_weight_path = './saved_models/model_adv_004_006_015.pth.tar'         

    if os.path.isfile(pre_weight_path):
        print("=> loading checkpoint '{}'".format(pre_weight_path))
        checkpoint = torch.load(pre_weight_path)
        #start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        print('Initiate accuracy: ', best_prec1)
        net.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format(pre_weight_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(pre_weight_path))


    optimizer = torch.optim.SGD(net.parameters(), lr=param['learning_rate'], momentum=param['momentum'],  weight_decay=param['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)


    if torch.cuda.is_available():
        print('CUDA enabled.')
        net.cuda()
    net.train()

    data_time_str = datetime.now().ctime()
    save_weights_dir = os.path.join('/data','adv_training_models',data_time_str)
    os.mkdir(save_weights_dir)

    write_para_info(param, PGD_param, filepath = os.path.join(save_weights_dir, 'para_info.txt'))

    loss_file = './loss_info_adv_v1_' + data_time_str + '.csv'
    with open(loss_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["loss_ori", "loss_pgd", "loss_cc", "loss_ori_eval", "loss_pgd_eval_mean"])
    print("Start training")
    # Train the model
    print('Abandoned initial accuracy. Set as 0.00')
    best_prec1 = 0.0
    for epoch in range(param['num_epochs']):
        net.train()
        loss_ori_list = []
        loss_pgd_list = []
        loss_cc_list = []
        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))
        for t, (x, y) in enumerate(tqdm(loader_train)):
            x_var, y_var = to_var(x), to_var(y.long())
            loss_ori = criterion(net(x_var), y_var)
            loss_ori_list.append(loss_ori.cpu().detach().numpy())
            '''
            optimizer.zero_grad()
            loss_ori.backward()
            optimizer.step()
            '''

            # adversarial training
            if epoch + 1 > param['delay'] and param['adv_PGD']:
                adversary = LinfPGDAttack(epsilon=PGD_param["epsilon"], k=PGD_param["k"], a=PGD_param["a"], random_start=PGD_param["random_start"])
                # use predicted label to prevent label leaking
                y_pred = pred_batch(x, net)
                x_adv = adv_train(x, y_pred, net, criterion, adversary)
                x_adv_var = to_var(x_adv)
                loss_pgd = criterion(net(x_adv_var), y_var)

                loss = (loss_ori + loss_pgd) / 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_pgd_list.append(loss_pgd.cpu().detach().numpy())

            if epoch + 1 > param['delay'] and param['adv_CC']:
                def wrapped_model(x_np):
                    x_np = x_np.transpose((0, 3, 1, 2))  # from NHWC to NCHW
                    x_t = torch.from_numpy(x_np).cuda()
                    net.eval()
                    with torch.no_grad():
                        result = net(x_t).cpu().numpy()
                    return result

                adversary = attacks.CommonCorruptionsAttack()
                x_adv = adversary(wrapped_model, x.numpy().transpose((0, 2, 3, 1)), y.numpy())
                x_adv = torch.from_numpy(x_adv.transpose((0, 3, 1, 2)))
                x_adv_var = to_var(x_adv)
                loss_cc = criterion(net(x_adv_var), y_var)
                loss_cc_list.append(loss_cc.cpu().detach().numpy())
            
                optimizer.zero_grad()
                loss_cc.backward()
                optimizer.step()

                
        lr_scheduler.step()

        if len(loss_ori_list) == 0:
            loss_ori_mean = 0
        else:
            loss_ori_mean = np.mean(np.array(loss_ori_list))

        if len(loss_pgd_list) == 0:
            loss_pgd_mean = 0
        else:
            loss_pgd_mean = np.mean(np.array(loss_pgd_list))

        if len(loss_cc_list) == 0:
            loss_cc_mean = 0
        else:
            loss_cc_mean = np.mean(np.array(loss_cc_list))

        # get evaluate loss
        loss_eval_list = []
        loss_pgd_eval_list = []
        for _, (x, y) in enumerate(tqdm(loader_val)):
            net.eval()
            x_var, y_var = to_var(x), to_var(y.long())
            output = net(x_var)
            loss_eval = criterion(output, y_var)
            loss_eval_list.append(loss_eval.cpu().detach().numpy())

            if epoch % 1 ==0 and param['adv_PGD']:
                adversary = LinfPGDAttack(epsilon=PGD_param["epsilon"], k=PGD_param["k"], a=PGD_param["a"], random_start=PGD_param["random_start"])
                y_pred = pred_batch(x, net)
                x_adv = adv_train(x, y_pred, net, criterion, adversary)
                x_adv_var = to_var(x_adv)
                loss_pgd_eval = criterion(net(x_adv_var), y_var)
                loss_pgd_eval_list.append(loss_pgd_eval.cpu().detach().numpy())

        loss_eval_mean = np.mean(np.array(loss_eval_list))
        if len(loss_pgd_eval_list) == 0:
            loss_pgd_eval_mean = 0
        else:
            loss_pgd_eval_mean = np.mean(np.array(loss_pgd_eval_list))

        print("loss_ori , loss_pgd , loss_cc, loss_ori_eval, loss_pgd_eval: ", loss_ori_mean, " , ", loss_pgd_mean, " , ", loss_cc_mean, " , ", loss_eval_mean, " , ", loss_pgd_eval_mean)
        # record loss info
        with open(loss_file, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([loss_ori_mean, loss_pgd_mean, loss_cc_mean, loss_eval_mean, loss_pgd_eval_mean])
        
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
        

        prec1 = 0.0
        is_best = prec1 >= best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
        'epoch': epoch + 1,
        'arch': 'RenNet50',
        'state_dict': net.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
        }, is_best, net, filename=os.path.join(save_weights_dir, 'undefended_pytorch_resnet_adv_' + str(epoch) + '.pth.tar'))

        #torch.save(net.state_dict(), 'state_dict_adv_'+ str(epoch) +'.pkl')

    #test(net, loader_val)

    #torch.save(net.state_dict(), 'state_dict_adv_final.pkl')

def write_para_info(param, PGD_param, filepath = './para_info.txt'):
    with open(filepath, 'w') as file:
        file.write("param: " + '\n')
        for key, value in param.items():
            file.write(key + " : " + str(value) + '\n')
        file.write('\n')
        file.write("PGDparam: " + '\n')
        for key, value in PGD_param.items():
            file.write(key + " : " + str(value) + '\n')


if __name__ == '__main__':
    main()