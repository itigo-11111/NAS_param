from numpy.core.fromnumeric import sort
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
# from decimal import Decimal, ROUND_DOWN
from decimal import Decimal, ROUND_HALF_UP
import logging
import random
from model import NetworkCIFAR as Network2

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    self.optimizer_gammas = torch.optim.SGD(self.model.arch_gammas_parameters(),
        lr=args.gammas_learning_rate, weight_decay=args.arch_weight_decay)
    self.auxiliary = args.auxiliary
    self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_gammas, float(args.epochs), eta_min=args.learning_rate_min)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer,device):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta),device)
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, pytorch_total_params_train,step,limit_param,num_flag,unrolled):
    self.optimizer.zero_grad()
    # print(self.model.alphas_normal)
    # print(self.model.gammas_normal)
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid,pytorch_total_params_train,step,limit_param,num_flag)
    self.optimizer.step()
    # print("after:")
    # print(self.model.alphas_normal)
    # print(self.model.gammas_normal)

  def param_step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, pytorch_total_params_train,step,limit_param, num_flag,lambda_a,param_prev):

    genotype = self.model.genotype()
    train_param = pytorch_total_params_train
    # if param_prev == pytorch_total_params_train: 
     

    self.model.gammas_normal.data = self.model.alphas_normal[:,4:].clone().detach()
    self.model.gammas_reduce.data = self.model.alphas_reduce[:,4:].clone().detach()

    aa = self.model.gammas_normal.clone().detach()

    
    # self.model.gammas_normal =  torch.tensor(self.model.gammas_normal + 4.0, requires_grad=True)
    # DropArchWeight
    if train_param == param_prev: 
      with torch.no_grad():
        # normal cell
        i = 0
        while True:    
          n = random.randint(0,13)
          num = torch.argmax(self.model.alphas_normal, dim=1)[n]
          if num > 3 or i > 20:
            break
          i += 1
        # self.model.alphas_normal[n][num] = torch.max(self.model.alphas_normal[n,:].clone().detach())
        sorted , _ = torch.sort(self.model.alphas_normal[n,:].clone().detach(),descending=True)
        self.model.alphas_normal[n][num] = sorted[2]

        # print("n:{} num:{}".format(n,num))

        # reduce cell
        i = 0
        while True:    
          n = random.randint(0,13)
          num = torch.argmax(self.model.alphas_reduce, dim=1)[n]
          if num > 3 or i > 20:
            break
          i += 1
        # self.model.alphas_reduce[n][num] = torch.min(self.model.alphas_reduce[n,:].clone().detach())  
        sorted , _ = torch.sort(self.model.alphas_reduce[n,:].clone().detach(),descending=True)
        self.model.alphas_reduce[n][num] = sorted[2]

      genotype = self.model.genotype()
      CIFAR_CLASSES = 10
      model3 = Network2(36, CIFAR_CLASSES, 20, self.auxiliary, genotype)
      train_param = sum(p.numel() for p in model3.parameters() if p.requires_grad)
      pytorch_total_params_train = train_param
      self.model.gammas_normal.data = self.model.alphas_normal[:,4:].clone().detach()
      self.model.gammas_reduce.data = self.model.alphas_reduce[:,4:].clone().detach()

    self.optimizer_gammas.zero_grad()

    # np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    logging.info(self.model.alphas_normal)

    # print(self.model.alphas_normal)

    # print(self.model.alphas_normal.size())
    # print(self.model.betas_reduce)
    # print(self.model.gammas_normal)


    # print("param:{} limit:{}".format(pytorch_total_params_train,limit_param)) 
    if pytorch_total_params_train > limit_param :
      self.param_backward_step(input_valid, target_valid, pytorch_total_params_train,step,limit_param,num_flag,lambda_a)
    
    
      self.optimizer_gammas.step()
      self.scheduler.step()
      # print(self.model.alphas_normal)
      # print(self.model.betas_normal)
      # print(self.model.gammas_normal)
      bb = self.model.gammas_normal.clone().detach()
      logging.info(aa - bb)
      # print(aa - bb)
      # print(self.model.alphas_reduce)


      gammas_normal_nonparam = self.model.alphas_normal[:,:4].clone().detach()
      gammas_reduce_nonparam = self.model.alphas_reduce[:,:4].clone().detach()
          
      self.model.alphas_normal.data = torch.cat([gammas_normal_nonparam,self.model.gammas_normal],dim=1)
      self.model.alphas_reduce.data = torch.cat([gammas_reduce_nonparam,self.model.gammas_reduce],dim=1)
    # print("after:")
    # print(self.model.alphas_normal)
    # print(self.model.betas_normal)
    # print(self.model.gammas_normal)
  
  # def _backward_step_old(self, input_valid, target_valid,pytorch_total_params_train,step,limit_param):
  #   loss = self.model._loss(input_valid, target_valid)
  #   loss.backward()
    return train_param,genotype

  def _backward_step(self, input_valid, target_valid,pytorch_total_params_train,step,limit_param,num_flag):
    loss = self.model._loss(input_valid, target_valid,num_flag)
    # print(loss.item())
    loss.backward()
    # print(input_valid.grad)
  
  def param_backward_step(self, input_valid, target_valid,pytorch_total_params_train,step,limit_param,num_flag,lambda_a):
    loss = self.model.param_loss(input_valid, target_valid,num_flag) 
    # loss = loss*0.0001 + torch.tensor(pytorch_total_params_train*lambda_a)
    # with torch.set_grad_enabled(False):
    loss = loss*torch.tensor((pytorch_total_params_train - limit_param),requires_grad=False)*lambda_a

    # loss = loss*0.00001 + torch.abs(torch.tensor(pytorch_total_params_train - limit_param))*lambda_a
    # loss = loss*0.00001
    # print("loss => {} param => {}".format(loss.item(),pytorch_total_params_train))
    logging.info("loss => {} param => {} ".format(loss.item(),pytorch_total_params_train))

    # loss.backward(retain_graph=True)
    loss.backward()
    # print(input_valid.grad)
    # print(Decimal(self.model.gammas_normal.grad).quantize(Decimal(.00000000???),rounding=ROUND_HALF_UP))
    # print("{0:.7f}".format(Decimal(self.model.gammas_normal.grad)))
    
    self.model.gammas_normal.grad = abs(self.model.gammas_normal.grad)
    self.model.gammas_reduce.grad = abs(self.model.gammas_reduce.grad)
    self.model.betas_normal.grad = abs(self.model.betas_normal.grad)
    self.model.betas_reduce.grad = abs(self.model.betas_normal.grad)

    # print(self.model.gammas_normal.grad)

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer,device):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer,device)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta,device):
    model_new = self.model.new(device)
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]
