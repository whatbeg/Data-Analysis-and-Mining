from __future__ import print_function
import numpy as np
import preprocessing as proc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='BASE Model')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--SKs', type=float, default=1.0, metavar='SK',
                    help='use how many channels times every layer (except last layer)')
parser.add_argument('--usedatasize', type=int, default=60000, metavar='SZ',
                    help='use how many training data to train network')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, int(20*args.SKs), kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.conv1_2 = nn.Conv2d(int(20*args.SKs), int(20*args.SKs), kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.conv2 = nn.Conv2d(int(20*args.SKs), int(50*args.SKs), kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.fc1 = nn.Linear(int(5*5*50*args.SKs), 500)
        self.fc2 = nn.Linear(500, 10)
        self.bn1_1 = nn.BatchNorm2d(int(20*args.SKs))
        self.bn1_2 = nn.BatchNorm2d(int(20*args.SKs))
        self.bn2 = nn.BatchNorm2d(int(50*args.SKs))
        self.bn3 = nn.BatchNorm1d(500)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(self.bn2(x), 2)
        x = x.view(-1, int(5*5*50*args.SKs))
        x = self.fc1(x)
        x = F.relu(self.bn3(x))
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
# print(model)
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


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
            yield proc.Normalize(data[start:end], (proc.TRAIN_MEAN,)*(end-start),
                                          (proc.TRAIN_STD,)*(end-start)), label[start:end]
        else:
            yield proc.Normalize(data[start:end], (proc.TRAIN_MEAN,)*(end-start),
                                          (proc.TRAIN_STD,)*(end-start)), label[start:end]


def train(epoch, train_data, train_labels, use_data_len=10000):
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
            print('Train Epoch: {} [{:5d}/{} ({:2d}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), use_data_len,
                int(100. * batch_idx * len(data) / use_data_len), loss.data[0]))
        batch_idx += 1


def test(test_data, test_labels):
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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, test_data.shape[0],
        100. * correct / test_data.shape[0]))


def go():
    train_images, train_labels = proc.get_data("train")
    test_images, test_labels = proc.get_data("test")
    for epoch in range(1, args.epochs + 1):
        train(epoch, train_images, train_labels, args.usedatasize)
        test(test_images, test_labels)

if __name__ == '__main__':
    go()
