import datetime
import time
import random
import os
import csv
import pickle

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchvision import transforms

from args import conf
from model_select import model_select
from train_val import train, val
from loadDB import DBLoader

args = conf()


def worker_init_fn(worker_id):
    random.seed(worker_id+args.seed)

test_transform = transforms.Compose([
                        transforms.Resize(args.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
test_dataset = DBLoader(args.path2db,'test',test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.num_workers,
                                        pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)


device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
cudnn.deterministic = True
# cudnn.benchmark = True
random.seed(args.seed)
torch.manual_seed(args.seed)


model = model_select(args)
model = torch.nn.DataParallel(model).to(device)

param = torch.load('./result/cifar10/NM/1/0123/cnn_cifar10_epoch_100.pth')
model.load_state_dict(param)

criterion = nn.CrossEntropyLoss()
test_loss, test_acc = val(args, model, device, test_loader, criterion)
print('Test Loss : {} \t Test Acc:{}'.format(test_loss, test_acc))