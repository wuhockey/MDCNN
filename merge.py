from .cnn import CNN
from .TextCNN import TextCNN
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

class MergedNN(nn.Module):

    def __init__(self):
        super(MergedNN, self).__init__()
        self.textnn = TextCNN()
        self.conv = CNN()
        self.class_num =2
        self.linear = nn.Linear(256,2)

    def forward(self, X1,X2):
      '''
      X: [batch_size, sequence_length]
      '''
      # print(X2)
      nlp = self.textnn(X2)
      cv = self.conv(X1)
      # print(nlp.shape)
      # print(cv.shape)
      output = self.linear(torch.cat((nlp,cv),dim=1)[:,:256])
      return output