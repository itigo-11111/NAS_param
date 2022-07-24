# -*- coding: utf-8 -*-
import sys
import datetime
import time
import random
import os
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import glob

from torchvision import transforms

from args import conf
from model_select import model_select
from train_val import train, val
from loadDB import DBLoader

args = conf()

def worker_init_fn(worker_id):
    random.seed(worker_id+args.seed)

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__== "__main__":
    args.seed = args.id *1000 + args.iteration
    print(args)

    start_time = time.time()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    #to deterministic
    cudnn.deterministic = True
    # cudnn.benchmark = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # args.numof_classes = len(glob.glob(args.path2db + 'train/*'))
    network_list = ['vgg16']

    for network in network_list:
        args.usenet = network    
        # Training settings
        img_size = (args.img_size_h, args.img_size_w)
        train_transform = transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.Pad(4, padding_mode = 'reflect'),
                            transforms.RandomCrop(img_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        train_dataset = DBLoader(args.path2db,'train',train_transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.num_workers,
                                                pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)
        
        # Validation settings
        val_transform = transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        val_dataset = DBLoader(args.path2db,'val',val_transform)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.num_workers,
                                                pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

        # test settings
        test_transform = transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        test_dataset = DBLoader(args.path2db,'test',test_transform)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.num_workers,
                                                pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

        # Model & optimizer
        model = model_select(args)
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        params = list(model.parameters())
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov = True)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=args.lr_decay)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
        # print(model)


        # create output directory
        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        # output_dir = './result/{0}/{1}/{2}/{3:04d}/'.format(\
                        # args.dataset, args.sampler, args.id, args.iteration)
        output_dir = './result/{}_{}/{}/'.format(args.usenet, args.dataset,now_time)
        create_dir(output_dir)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("number of parameters : {} (trainable only : {})".format(pytorch_total_params,pytorch_total_params_train))
        
        with open(output_dir+'param.txt', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['param_size', 'param_size_train'])
            writer.writerow([pytorch_total_params,pytorch_total_params_train])

        # Training & Validation
        csv_file_path = output_dir + 'output.csv'
        with open(csv_file_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['train_loss', 'train_acc', 'val_loss', 'val_acc', 'time','lr' , 'epoch','val_time','latency'])
        best_acc , best_train_acc = 0 , 0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train(args, model, device, train_loader, optimizer, epoch, criterion)
            scheduler.step()
            val_loss, val_acc,latency, val_time = val(args, model, device, val_loader, criterion)

            if train_acc > best_train_acc:
                best_train_acc = train_acc
            if val_acc > best_acc:
                best_acc = val_acc
            now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            print('Epoch[{0:03}/{1:03}]  Train Loss:{2:.6f} Train Acc:{3:.6f} Best Train Acc:{4:.6f} Val Loss:{5:.6f} Val Acc:{6:.6f} Best Val Acc : {7:.6f} Val time:{8:.6f} Latency:{9:.6} Now {10} '.format(\
                        epoch, args.epochs, train_loss, train_acc, best_train_acc, val_loss, val_acc, best_acc,  val_time, latency, now_time))

            if epoch % args.save_interval == 0:
                saved_weight = output_dir + '{}_{}_epoch_{}.pth'.format(args.usenet, args.dataset, epoch)
                torch.save(model.cpu().state_dict(), saved_weight)
                model_state = model.cpu().state_dict()
                model = model.to(device)
            with open(csv_file_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([train_loss, train_acc, val_loss, val_acc, now_time,optimizer.param_groups[0]['lr'], epoch,val_time,latency])

        # test_loss, test_acc = val(args, model, device, test_loader, criterion)
        # with open(output_dir+'test.csv', 'w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['test_loss', 'test_acc'])
        #     writer.writerow([test_loss, test_acc])
        with open(output_dir + 'info.pickle', 'wb') as f:
            pickle.dump(args, f)



        end_time = time.time()
        interval = end_time - start_time
        interval = str("time = %dh %dm %ds" % (int(interval/3600),int((interval%3600)/60),int((interval%3600)%60)))
        with open(output_dir+'time.txt', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['time'])
            writer.writerow([interval])