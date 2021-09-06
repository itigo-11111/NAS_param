import os
import sys
import time as tm
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import datetime
import csv
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from collections import namedtuple
import train as training

from torch.autograd import Variable
from model_search import Network
from model import NetworkCIFAR as Network2

from architect import Architect
from torchvision import transforms
import random
from tqdm import tqdm

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='./cifar10/', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# 50
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save_dir', type=str, default='EXP', help='experiment name')
# parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument("--num_workers", default=8, type=int, help="num of workers(data_loader)")
parser.add_argument('--multigpu', default=True, action='store_true', help='If true, training is not performed.')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--train_mode', action='store_true', default=False, help='use train after search')
parser.add_argument('--val_mode', action='store_true', default=False, help='use validation and check accuracy')
parser.add_argument("--seed", default=1, type=int, help="seed")
parser.add_argument("--iteration", default=1, type=int, help="iteration")
parser.add_argument("--id", default=1, type=int, help="sampler id")

args = parser.parse_args()
args.img_size = (32, 32)

def main():
  for id in range(2,12):
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    args.iteration = id
    args.seed = args.id *10000 + args.iteration
    # args.save = './result/search-{}_{}/{}/'.format(args.save,args.set, tm.strftime("%Y%m%d-%H%M%S"))
    args.save = './result_param/search-{}_{}/{}/'.format(args.save_dir,args.set,args.seed)
    create_dir(args.save)
    # utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    CIFAR_CLASSES = 10
    if args.set=='cifar100':
        CIFAR_CLASSES = 100

    if not torch.cuda.is_available():
      logging.info('no gpu device available')
      sys.exit(1)

    random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    # cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    # torch.cuda.manual_seed(args.seed)
    # logging.info('gpu device = %d' % args.gpu)
    # logging.info("args = %s", args)
    start_time = tm.time()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.to(device)
    # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    def worker_init_fn(worker_id):
      random.seed(worker_id+args.seed)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # train_transform, valid_transform = utils._data_transforms_cifar10(args)
    # if args.set=='cifar100':
    #     train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    # else:
    #     train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    # num_train = len(train_data)
    # indices = list(range(num_train))
    # split = int(np.floor(args.train_portion * num_train))

    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batch_size,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    #     pin_memory=True, num_workers=2)


    # valid_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=args.batch_size,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
    #     pin_memory=True, num_workers=2)

    train_transform = transforms.Compose([
                        transforms.Resize(args.img_size),
                        transforms.Pad(4, padding_mode = 'reflect'),
                        transforms.RandomCrop(args.img_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        # transforms.Normalize((-0.0891, 0.0698, 0.3051), (1.1908, 1.1972, 1.1822))])
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_fractal = dset.ImageFolder(os.path.join(args.data, 'train'),transform=train_transform)
    # train_fractal = DBLoader(args.data,'TRAIN',train_transform)
    train_queue = torch.utils.data.DataLoader(dataset=train_fractal, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.num_workers,
                                            pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn)

    val_transform = transforms.Compose([
                      transforms.Resize(args.img_size),
                      transforms.ToTensor(),
                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # val_dataset = DBLoader(args.path2db,'VALIDATION',val_transform)
    val_dataset = dset.ImageFolder(os.path.join(args.data, 'val'),transform=val_transform)
    valid_queue = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers,
                                            pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    csv_file_path = args.save + 'output.csv'
    with open(csv_file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['train_acc','train_loss',  'time', 'lr' , 'epoch'])
    
    csv_file_path_param = args.save + 'param.csv'
    with open(csv_file_path_param, 'w') as f:
      writer = csv.writer(f)
      writer.writerow(['param_size','latency', 'val_time', 'val_acc' , 'epoch','model','param_train'])
    best_acc = 0.0
    best_train_acc = 0.0

    for epoch in range(1, args.epochs + 1):
      if epoch != 1:
        scheduler.step()
      lr = scheduler.get_last_lr()[0]
      # logging.info('epoch %d lr %e', epoch, lr)

      genotype = model.genotype()
      # logging.info('genotype = %s', genotype)

      #print(F.softmax(model.alphas_normal, dim=-1))
      #print(F.softmax(model.alphas_reduce, dim=-1))

      # training
      train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch,device)
      # logging.info('train_acc %f', train_acc)

      
      if train_acc > best_train_acc:
        best_train_acc = train_acc
        if not args.val_mode:
          best_genotype = genotype

      time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
      with open(csv_file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([train_acc, train_obj,  time, optimizer.param_groups[0]['lr'], epoch])


      # validation
      if not args.val_mode:
        if args.epochs-epoch<=1:
          valid_acc, valid_obj,latency,val_time = infer(valid_queue, model, criterion,device)
        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print('Epoch[{0:03}/{1:03}]  Train Loss:{2:.6f} Train Acc:{3:.6f} Best Train Acc:{4:.6f}  Now {5}'.format(\
                      epoch, args.epochs, train_obj, train_acc, best_train_acc, now_time))
        training.main(genotype,args.seed,epoch)

      else:
        valid_acc, valid_obj,latency,val_time = infer(valid_queue, model, criterion,device)
        if valid_acc > best_acc:
          best_acc = valid_acc
          best_genotype = genotype


        model2 = Network2(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
        pytorch_total_params = sum(p.numel() for p in model2.parameters())
        pytorch_total_params_train = sum(p.numel() for p in model2.parameters() if p.requires_grad)

        # pytorch_total_params = sum(p.numel() for p in model.parameters())
        # pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)


        # logging.info('number of parameters : %s (trainable only : %s)',pytorch_total_params,pytorch_total_params_train)
        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        print('Epoch[{0:03}/{1:03}]  Train Loss:{2:.6f} Train Acc:{3:.6f} Best Train Acc:{4:.6f} Val Loss:{5:.6f} Val Acc:{6:.6f} Best Val Acc : {7:.6f} Val time:{8:.6f} Latency:{9:.6} Num of Param:{10} Now {11} '.format(\
                      epoch, args.epochs, train_obj, train_acc, best_train_acc, valid_obj, valid_acc, best_acc, val_time, latency, pytorch_total_params_train, now_time))
        with open(csv_file_path_param, 'a') as f:
          writer = csv.writer(f)
          writer.writerow([pytorch_total_params,latency,val_time,valid_acc,epoch,genotype, pytorch_total_params_train])


      utils.save(model, os.path.join(args.save, 'weights.pt'))
    end_time = tm.time()
    interval = end_time - start_time
    interval = str("time = %dh %dm %ds" % (int(interval/3600),int((interval%3600)/60),int((interval%3600)%60)))
    with open(args.save+'time.txt', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['time'])
        writer.writerow([interval])

    if args.train_mode:
      # val_mode ON -> use best val_acc  OFF -> use best train_acc
      training.main(best_genotype)

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch,device):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  bar = tqdm(desc = "Training", total = len(train_queue), leave = False)
  i = valid_queue
  for step, (input, target) in enumerate(train_queue):
    model.train()

    n = input.size(0)
    input = Variable(input, requires_grad=False).to(device)
    target = Variable(target, requires_grad=False).to(device)

    # get a random minibatch from the search queue with replacement
    # input_search, target_search = next(i)
    try:
     input_search, target_search = next(valid_queue_iter)
    except:
     valid_queue_iter = iter(valid_queue)
     input_search, target_search = next(valid_queue_iter)
    
    input_search = Variable(input_search, requires_grad=False).to(device)
    target_search = Variable(target_search, requires_grad=False).to(device)

    # epochs >= 15 -> 0
    if epoch>=0:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    bar.set_description("Loss: {0:.6f}, Accuracy: {1:.6f}".format(objs.avg, top1.avg))
    bar.update()
    # if step % args.report_freq == 0:
      # logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  bar.close()
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion,device):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  latency_flag = 0
  start = tm.time()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      if latency_flag == 0:
        start_latency = tm.time()

      #input = input.cuda()
      #target = target.cuda(non_blocking=True)
      input = Variable(input).to(device)
      target = Variable(target).to(device)
      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if latency_flag == 0:
        latency = tm.time() - start_latency
        latency_flag = 1      

      # if step % args.report_freq == 0:
      #   logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    elapsed_time = tm.time() - start
    return top1.avg, objs.avg,latency*1000,elapsed_time


if __name__ == '__main__':
  main() 

