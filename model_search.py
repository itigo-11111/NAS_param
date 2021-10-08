import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES,PRIMITIVES_PARAM
from genotypes import Genotype
import numpy as np

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedOp(nn.Module):

  def __init__(self, C, stride,num_flag):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2,2,ceil_mode=True) # modify 7/14
    self.k = 4
    self.num_flag = num_flag
    for primitive in PRIMITIVES:
      op = OPS[primitive](C //self.k, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C //self.k, affine=False))
      self._ops.append(op)


  def forward(self, x, weights,num_flag,param_gate):
    #channel proportion k=4
    dim_2 = x.shape[1]

    if num_flag == 1:
      # param optimization
      self.k = 4
      xtemp = x[ : , :  dim_2//self.k, :, :]
      xtemp2 = x[ : ,  dim_2//self.k:, :, :]

      # temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
      i  = 0
      temp1 = 0.0
      for w, op in zip(weights, self._ops[4:]):
        # if param_gate[i].item() == 1:
        #   temp1 += w*op(xtemp)
        temp1 += w*op(xtemp)
        # print(op)
        i += 1
      # temp1 = torch.tensor(temp1, dtype=torch.float)
      # print(self._ops)
      # print("x.shape={} x.shape[1]={} x.shape[2]={}".format(x.shape,x.shape[1],x.shape[2]))
      # print("num of w=>{} num of ops => {}".format(len(weights),len(self._ops)))
      # for w, op in zip(weights, self._ops):
      #   print(op)
      
    else:
      self.k = 4
      xtemp = x[ : , :  dim_2//self.k, :, :]
      xtemp2 = x[ : ,  dim_2//self.k:, :, :]
      temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
    #reduction cell needs pooling before concat

    if temp1.shape[2] == x.shape[2]:

      ans = torch.cat([temp1,xtemp2],dim=1)
    else:
      # print('{} \t {}'.format(temp1.shape,self.mp(xtemp2).shape))
      ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
    ans = channel_shuffle(ans,self.k)
    # ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
    #except channe shuffle, channel shift also works
    return ans


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, num_flag):
    super(Cell, self).__init__()
    self.reduction = reduction
    self.num_flag = num_flag

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()

    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride,num_flag)
        self._ops.append(op)

  def forward(self, s0, s1, weights,weights2,num_flag,param_gate):
     # weights => 14*8 or 14*4  weights => 14(2-3-4-5)
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(weights2[offset+j]*self._ops[offset+j](h, weights[offset+j],num_flag,param_gate) for j, h in enumerate(states))
      
      offset += len(states)
      states.append(s)
    return torch.cat(states[-self._multiplier:], dim=1) # concut outputs without s0,s1


class Network(nn.Module):

  def __init__(self,num_flag, device, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.device = device
    self.num_flag = num_flag

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, num_flag)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
    self.classifier = nn.Linear(C_prev, num_classes)
 
    self._initialize_alphas(device)

  def new(self,device):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).to(device)
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input,num_flag):
    s0 = s1 = self.stem(input)
    # if num_flag == 1:
    #   # with torch.no_grad():
    #   self.gammas_normal = self.alphas_normal[:,4:]
    #   self.gammas_reduce = self.alphas_reduce[:,4:]
    #   print(self.alphas_normal)
    #   print(self.gammas_normal)
    # update weights => concat normal and normal_param 

    for i, cell in enumerate(self.cells):
      if cell.reduction:
        # with torch.no_grad():
        if num_flag == 1:
          # weights = F.softmax(self.alphas_reduce[:,4:], dim=-1)
          weights = F.softmax(self.gammas_reduce, dim=-1)
        else:
          weights = F.softmax(self.alphas_reduce, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        for i in range(self._steps-1):
          end = start + n
          tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      else:
        # with torch.no_grad():
        if num_flag == 1:
          # weights = F.softmax(self.alphas_normal[:,4:], dim=-1)
          weights = F.softmax(self.gammas_normal, dim=-1)
        else:
          weights = F.softmax(self.alphas_normal, dim=-1)
        n = 3
        start = 2
        weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
        for i in range(self._steps-1):
          end = start + n
          tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
          start = end
          n += 1
          weights2 = torch.cat([weights2,tw2],dim=0)
      # print(weights2)
      # print(weights2.size())
      s0, s1 = s1, cell(s0, s1, weights,weights2,num_flag,self.param_gate)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target,num_flag):
    logits = self(input,num_flag)
    return self._criterion(logits, target) 

  def param_loss(self, input, target,num_flag):
    logits = self(input,num_flag)
    return self._criterion(logits, target) 

  def _initialize_alphas(self,device):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    # all operations
    num_ops = len(PRIMITIVES)
    # learnable operations
    num_ops_param = len(PRIMITIVES_PARAM)

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops).to(device), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops).to(device), requires_grad=True)
    self.gammas_normal = Variable(1e-3*torch.randn(k, num_ops_param).to(device), requires_grad=True)
    self.gammas_reduce = Variable(1e-3*torch.randn(k, num_ops_param).to(device), requires_grad=True)
    self.betas_normal = Variable(1e-3*torch.randn(k).to(device), requires_grad=True)
    self.betas_reduce = Variable(1e-3*torch.randn(k).to(device), requires_grad=True)
    self.param_gate = Variable(torch.tensor([0,0,0,0,1,1,1,1]).to(device), requires_grad=False)
    #param_gate is binary gate : none ,max_poll,avg_pool,skip_connect(weight-free) => 0, sep_conv_x,dil_conv=> 1 
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
      self.betas_normal,
      self.betas_reduce,
      self.param_gate
    ]
    self._arch_gammas_parameters = [
      self.gammas_normal,
      self.gammas_reduce,
      self.betas_normal,
      self.betas_reduce,
    ]
    # print(self.betas_normal)

  def arch_parameters(self):
    return self._arch_parameters

  def arch_gammas_parameters(self):
    return self._arch_gammas_parameters

  def genotype(self):

    def _parse(weights,weights2):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        W2 = weights2[start:end].copy()
        for j in range(n):
          W[j,:]=W[j,:]*W2[j]
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        
        #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    n = 3
    start = 2
    weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
    weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
    # print(weightsn2)
    for i in range(self._steps-1):
      end = start + n
      tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
      tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
      start = end
      n += 1
      weightsr2 = torch.cat([weightsr2,tw2],dim=0)
      weightsn2 = torch.cat([weightsn2,tn2],dim=0)
    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().detach().numpy(),weightsn2.data.cpu().detach().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().detach().numpy(),weightsr2.data.cpu().detach().numpy())
    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

