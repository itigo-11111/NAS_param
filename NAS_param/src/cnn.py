import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, args.output_ch_1, kernel_size=3)
        self.conv2 = nn.Conv2d(args.output_ch_1, args.output_ch_2, kernel_size=3)
        self.conv3 = nn.Conv2d(args.output_ch_2, args.output_ch_3, kernel_size=3)
        
        self.dropout1 = nn.Dropout(args.dropout_rate_1)
        self.dropout2 = nn.Dropout(args.dropout_rate_2)

        self.fc1 = nn.Linear(args.output_ch_3*2*2, args.output_fc_ch_1)
        self.fc2 = nn.Linear(args.output_fc_ch_1, args.output_fc_ch_2)
        self.fc3 = nn.Linear(args.output_fc_ch_2, args.numof_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim = 1)

def cnn3(args):
    return CNN(args)