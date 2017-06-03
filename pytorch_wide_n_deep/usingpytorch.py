from __future__ import print_function
import numpy as np
import dataprocessing as proc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='BASE Model')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.emb1 = nn.Embedding(9, 8, 0)
        self.emb2 = nn.Embedding(16, 8, 0)
        self.emb3 = nn.Embedding(2, 8, 0)
        self.emb4 = nn.Embedding(6, 8, 0)
        self.emb5 = nn.Embedding(42, 8, 0)
        self.emb6 = nn.Embedding(15, 8, 0)
        self.linear1 = nn.Linear(53, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(57, 2)

    def forward(self, x):
        wide_indices = Variable(torch.LongTensor([0, 1, 2, 3, 4, 5, 6]))
        wide = torch.index_select(x, 1, wide_indices).float()
        deep_indices = Variable(torch.LongTensor([16, 17, 18, 19, 20]))
        x1 = self.emb1(x.select(1, 10))
        x2 = self.emb2(x.select(1, 11))
        x3 = self.emb3(x.select(1, 12))
        x4 = self.emb4(x.select(1, 13))
        x5 = self.emb5(x.select(1, 14))
        x6 = self.emb6(x.select(1, 15))
        x7 = torch.index_select(x.float(), 1, deep_indices).float()
        deep = Variable(torch.cat([x1.data, x2.data, x3.data, x4.data, x5.data, x6.data, x7.data], 1))
        deep = F.relu(self.linear1(deep))
        deep = F.relu(self.linear2(deep))
        x = Variable(torch.cat([wide.data, deep.data], 1))
        x = self.linear3(x)
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def generate_data(data, label, batchSize, data_type='train', shuffle=True):
    assert batchSize > 0
    data_len = data.shape[0]
    total_batch = data_len / batchSize + (1 if data_len % batchSize != 0 else 0)
    if shuffle:
        indices = np.random.permutation(data_len)
        data = data[indices]
        label = label[indices]
    for idx in range(total_batch):
        start = idx * batchSize
        end = min((idx + 1) * batchSize, data_len)
        if data_type == 'train':
            yield data[start:end], label[start:end]
        else:
            yield data[start:end], label[start:end]


def train(epoch, train_data, train_labels, use_data_len=32561):
    model.train()   # set to training mode
    batch_idx = 1
    for (_data, _target) in generate_data(train_data[:use_data_len], train_labels[:use_data_len], batchSize=args.batch_size, shuffle=True):
        data = torch.from_numpy(_data)
        target = torch.from_numpy(_target).long()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [Iteration {}] [{:5d}/{} ({:2d}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, batch_idx * len(data), use_data_len,
                int(100. * batch_idx * len(data) / use_data_len), loss.data[0]))
        batch_idx += 1


def test(epoch, test_data, test_labels):
    model.eval()   # set to evaluation mode
    test_loss = 0
    correct = 0
    for (data, target) in generate_data(test_data, test_labels,
                                          batchSize=args.batch_size, shuffle=True):
        data = torch.from_numpy(data)
        target = torch.from_numpy(target).long()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model.forward(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= test_data.shape[0]  # loss function already averages over batch size
    print('\nEpoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        epoch, test_loss, correct, test_data.shape[0], 100. * correct / test_data.shape[0]))


def go():
    train_data, train_labels = proc.get_data("train")
    test_data, test_labels = proc.get_data("test")
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_data, train_labels, 32561)
        test(epoch, test_data, test_labels)

if __name__ == '__main__':
    go()
