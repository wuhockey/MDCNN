import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


embedding_size = 2
vocab_size = 256

class TextCNN(nn.Module):

    def __init__(self,base_line = False):
        super(TextCNN, self).__init__()
        self.base_line = base_line
        self.tokenizer = BertTokenizer.from_pretrained('././bert-base-uncased')
        self.vocab_size = len(self.tokenizer)
        self.W = nn.Embedding(self.vocab_size, embedding_size)
        output_channel = 3
        self.conv = nn.Sequential(nn.Conv2d(1, output_channel, (2,embedding_size)),
                                nn.ReLU(),
                                nn.MaxPool2d((2,1)))
        self.linear = nn.Linear(300,2)

    def forward(self, X):
      '''
      X: [batch_size, sequence_length]
      '''
      temp_list = []
      # X=torch.tensor(self.tokenizer(X,max_length=256,padding='max_length')['input_ids'],dtype=torch.long)
      batch_size = len(X)
      embedding_X = self.W(X) # [batch_size, sequence_length, embedding_size]
      embedding_X = embedding_X.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
      conved = self.conv(embedding_X) # [batch_size, output_channel,1,1]
      flatten = conved.view(batch_size, -1)# [batch_size, output_channel*1*1]
      if self.base_line:
          flatten = self.linear(flatten[:,0:300])
      return flatten

if __name__ == '__main__':
    textnn = TextCNN()
    textnn([['hello','my''name'],['hello','my''name']])