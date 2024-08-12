import torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self,base_line=False):
        super(CNN, self).__init__()
        self.base_line = base_line
        self.conv_1 = nn.Conv2d(in_channels=2,out_channels=8,kernel_size=(2,2))
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 2))
        self.linear = nn.Linear(30000,2)
    def forward(self, X):
      '''
      X: [batch_size, sequence_length]
      '''
      output = self.conv_1(X)
      output = self.pool(output)
      output = self.conv_2(output)
      output =self.pool(output)
      output = output.reshape(output.shape[0],-1)
      if self.base_line:
          if output.shape[1]<30000:
              temp_tensor = torch.zeros(size=(output.shape[0],30000-output.shape[1])).cuda()
              output = torch.hstack((output.cuda(),temp_tensor.cuda())).cuda()
          output = self.linear(output[:,0:30000])

      return output

if __name__ == '__main__':
     temp_tensor = torch.FloatTensor(1,8, 13, 12724)
     cnn = CNN()
     result_tensor = cnn(temp_tensor)