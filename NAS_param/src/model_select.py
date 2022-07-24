# -*- coding: utf-8 -*-

import os
import sys
import glob

import torch
import torch.nn as nn
import torchvision


from cnn import *
from resnet import *
from vgg import *
from shufflenetv2 import *
from wide_resnet import *
def model_select(args):

    args.numof_classes = len(glob.glob(args.path2db + 'train/*'))
    print(args.numof_classes)
    # convolutional neural network (3-conv)
    if args.usenet == 'cnn':
        model = cnn3(args)
        model.fc3 = nn.Linear(model.fc3.in_features, args.numof_classes)

    # Residual Network
    # https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
    elif args.usenet == 'resnet18':
        print('Select model : {}'.format(args.usenet))
        model = resnet18(args)
        model.fc= nn.Linear(model.fc.in_features, args.numof_classes)
    if args.usenet == 'resnet34':
        print('Select model : {}'.format(args.usenet))
        model = resnet34(args)
        model.fc= nn.Linear(model.fc.in_features, args.numof_classes)
    elif args.usenet == 'resnet50':
        print('Select model : {}'.format(args.usenet))
        model = resnet50(args)
        model.fc= nn.Linear(model.fc.in_features, args.numof_classes)
    elif args.usenet == 'resnet101':
        print('Select model : {}'.format(args.usenet))
        model = resnet101(args)
        model.fc= nn.Linear(model.fc.in_features, args.numof_classes)
    elif args.usenet == 'resnet152':
        print('Select model : {}'.format(args.usenet))
        model = resnet152(args)
        model.fc= nn.Linear(model.fc.in_features, args.numof_classes)
    
    # Wide Residual Network
    # http://www.bmva.org/bmvc/2016/papers/paper087/index.html
    elif args.usenet == 'wideresnet':
        print('Select model : {}'.format(args.usenet))
        model = wideresnet(args)

    # Visual Geometry Group(VGG) Network
    # https://www.robots.ox.ac.uk/~vgg/publications/2015/Simonyan15/
    elif args.usenet == 'vgg16':
        print('Select model : {}'.format(args.usenet))
        model = vgg16_bn(args)
        model.classifier[6] = nn.Linear(4096, args.numof_classes)
    
    # ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    # https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html

    elif args.usenet == 'shufflenet_v2_x0_5':
        print('Select model : {}'.format(args.usenet))
        model = shufflenet_v2_x0_5(args)
        model.fc= nn.Linear(model.fc.in_features, args.numof_classes)
    elif args.usenet == 'shufflenet_v2_x1_0':
        print('Select model : {}'.format(args.usenet))
        model = shufflenet_v2_x1_0(args)
        model.fc= nn.Linear(model.fc.in_features, args.numof_classes)
    elif args.usenet == 'shufflenet_v2_x1_5':
        print('Select model : {}'.format(args.usenet))
        model = shufflenet_v2_x1_5(args)
        model.fc= nn.Linear(model.fc.in_features, args.numof_classes)        
    elif args.usenet == 'shufflenet_v2_x2_0':
        print('Select model : {}'.format(args.usenet))
        model = shufflenet_v2_x2_0(args)
        model.fc= nn.Linear(model.fc.in_features, args.numof_classes)



    return model
