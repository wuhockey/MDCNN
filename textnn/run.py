import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F

dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

embedding_size = 2
sequence_length = len(sentences[0])
num_classes = len(set(labels))
batch_size = 3

word_list = " ".join(sentences).split()
vocab = list(set(word_list))
word2idx = {w:i for i,w in enumerate(vocab)}
vocab_size = len(vocab)

def make_data(sentences, labels):
    inputs = []
    for sen in sentences:
        inputs.append([word2idx[n] for n in sen.split()])

    targets = []
    for out in labels:
        targets.append(out)

    return inputs, targets

input_batch, target_batch = make_data(sentences, labels)
input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(target_batch)

dataset = Data.TensorDataset(input_batch,target_batch)
loader = Data.DataLoader(dataset, batch_size, True)

class TextCNN(nn.Module):

    def __init__(self):
        super(TextCNN, self).__init__()
        self.W = nn.Embedding(vocab_size, embedding_size)
        output_channel = 3
        self.conv = nn.Sequential(nn.Conv2d(1, output_channel, (2,embedding_size)),
                                nn.ReLU(),
                                nn.MaxPool2d((2,1)))
        self.fc = nn.Linear(output_channel,num_classes)

    def forward(self, X):
      '''
      X: [batch_size, sequence_length]
      '''
      batch_size = X.shape[0]
      embedding_X = self.W(X) # [batch_size, sequence_length, embedding_size]
      embedding_X = embedding_X.unsqueeze(1) # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
      conved = self.conv(embedding_X) # [batch_size, output_channel,1,1]
      flatten = conved.view(batch_size, -1)# [batch_size, output_channel*1*1]
      output = self.fc(flatten)
      return output

model = TextCNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(5000):
  for batch_x, batch_y in loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    pred = model(batch_x)
    loss = criterion(pred, batch_y)
    if (epoch + 1) % 1000 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

test_text = 'sorry for that'
tests = [[word2idx[n] for n in test_text.split()]]
test_batch = torch.LongTensor(tests).to(device)
# Predict
model = model.eval()
predict = model(test_batch).data.max(1, keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text,"is Bad Mean...")
else:
    print(test_text,"is Good Mean!!")
