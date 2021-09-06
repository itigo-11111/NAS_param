# -*- coding: utf-8 -*-

import argparse

from numpy import deg2rad

def conf():
    parser = argparse.ArgumentParser(description="PyTorch Deep Learning")
    
    # model name
    parser.add_argument("--dataset", default="cifar10", type = str, help="training dataset name")
    
    # paths
    parser.add_argument("--path2db", default="./cifar10/", type = str, help="path to training dataset")
    
    
    # network settings
    parser.add_argument("--usenet", default="shufflenet_v2_x0_5", type = str, help="use network")
    parser.add_argument("--epochs", default=600, type = int, help="end epoch")
    parser.add_argument("--numof_classes", default=10, type=int, help = 'number of classes')
    
    # hyperparameters
    parser.add_argument("--lr", default=0.025, type = float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type = float, help="momentum")
    parser.add_argument("--weight_decay", default=3e-4, type = float, help="weight decay")
    parser.add_argument("--batch_size", default=128, type = int, help="input batch size for training")
    parser.add_argument("--lr_decay", default=0.2, type=float, help="learning rate decay")

    parser.add_argument("--output_ch_1", default=64, type=int, help="conv1 output channel")
    parser.add_argument("--output_ch_2", default=64, type=int, help="conv2 output channel")
    parser.add_argument("--output_ch_3", default=128, type=int, help="conv3 output channel")
    
    
    #CNN
    parser.add_argument("--dropout_rate_1", default=0.3, type=float, help="dropout rate")
    parser.add_argument("--dropout_rate_2", default=0.3, type=float, help="dropout rate")
    parser.add_argument("--output_fc_ch_1", default=64, type=int, help="conv1 output channel")
    parser.add_argument("--output_fc_ch_2", default=64, type=int, help="conv1 output channel")


    # ResNet
    parser.add_argument("--output_ch_4", default=256, type=int, help="conv4 output channel")
    parser.add_argument("--output_ch_5", default=512, type=int, help="conv5 output channel")

    # WideResNet
    parser.add_argument("--width_coef1", default=10, type=int, help="width coef for WideResNet")
    parser.add_argument("--width_coef2", default=10, type=int, help="width coef for WideResNet")
    parser.add_argument("--width_coef3", default=10, type=int, help="width coef for WideResNet")
    parser.add_argument("--n_blocks1", default=4, type=int, help="brocks1 for WideResNet")
    parser.add_argument("--n_blocks2", default=4, type=int, help="brocks1 for WideResNet")
    parser.add_argument("--n_blocks3", default=4, type=int, help="brocks1 for WideResNet")
    
    parser.add_argument("--dropout_rate_3", default=0.3, type=float, help="dropout rate for WideResNet")

    
    # etc
    parser.add_argument("--img_size_w", default=32, type=int, help="image size width")
    parser.add_argument("--img_size_h", default=32, type=int, help="image size height")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers(data_loader)")
    parser.add_argument("--log_interval", default=25, type=int, help="how many batches to wait before logging training status")
    parser.add_argument("--save_interval", default=20, type = int, help="save every N epoch")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument("--iteration", default=1, type=int, help="iteration")
    parser.add_argument("--id", default=1, type=int, help="sampler id")
    parser.add_argument("--sampler", default='hoge', type=str, help="sampler name")
    args = parser.parse_args()
    return args
