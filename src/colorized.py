import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(483)

x_train_np = np.load('train.npy')
y_train_np = np.load('train_gt.npy')

x_valid_np = np.load('valid.npy')
y_valid_np = np.load('valid_gt.npy')

x_test_np = np.load('test.npy')

x_train_np = np.transpose(x_train_np, (0, 3, 1, 2))
y_train_np = np.transpose(y_train_np, (0, 3, 1, 2))

x_valid_np = np.transpose(x_valid_np, (0, 3, 1, 2))
y_valid_np = np.transpose(y_valid_np, (0, 3, 1, 2))

x_test_np = np.transpose(x_test_np, (0, 3, 1, 2))

x_train = Variable(torch.from_numpy(x_train_np)).float()
y_train = Variable(torch.from_numpy(y_train_np)).float()

x_valid = Variable(torch.from_numpy(x_valid_np)).float()
y_valid = Variable(torch.from_numpy(y_valid_np)).float()

x_test = Variable(torch.from_numpy(x_test_np)).float()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=16),
            torch.nn.MaxPool2d(kernel_size=4, stride=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(num_features=24),
            torch.nn.MaxPool2d(kernel_size=4, stride=2)
        )
        self.conv = torch.nn.Conv2d(in_channels=24, out_channels=2, kernel_size=3, stride=1, padding=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.conv(x)
        return x

epochs = 200
batch_size = 1
loss_train_arr = []
loss_valid_arr = []

net = Net()
net = net.cuda()
optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-4)
loss_func = torch.nn.MSELoss()

for i in range(epochs):

    sum_loss = 0
    for idx in range(int(800/batch_size)):

        x_train_ = x_train[idx*batch_size : (idx+1)*batch_size, :, :, :]
        y_train_ = y_train[idx*batch_size : (idx+1)*batch_size, :, :, :]

        x_train_batch = x_train_.cuda()
        y_train_batch = y_train_.cuda()

        y_pred_train = net(x_train_batch)
        loss_train = loss_func(y_pred_train, y_train_batch)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        sum_loss = sum_loss + loss_train.cpu().detach().numpy()

    sum_loss = sum_loss / int(800 / batch_size)
    loss_train_arr.append(sum_loss)
    print('epoch ', i , ' loss train: ', sum_loss)

    sum_loss = 0
    for idx in range(int(100/batch_size)):

        x_valid_ = x_valid[idx*batch_size : (idx+1)*batch_size, :, :, :]
        y_valid_ = y_valid[idx*batch_size : (idx+1)*batch_size, :, :, :]

        x_valid_batch = x_valid_.cuda()
        y_valid_batch = y_valid_.cuda()

        y_pred_valid = net(x_valid_batch)
        loss_valid = loss_func(y_pred_valid, y_valid_batch)

        sum_loss = sum_loss + loss_valid.cpu().detach().numpy()

    sum_loss = sum_loss / int(100/batch_size)
    loss_valid_arr.append(sum_loss)
    print('epoch ', i, ' loss valid: ', sum_loss)

y_pred_valid = np.ndarray(shape=(100, 2, 64, 64))
for idx in range(int(100/batch_size)):
    x_valid_ = x_valid[idx * batch_size: (idx + 1) * batch_size, :, :, :]

    x_valid_batch = x_valid_.cuda()
    y_pred_valid_ = net(x_valid_batch)

    y_pred_valid[idx * batch_size: (idx + 1) * batch_size, :, :, :] = y_pred_valid_.cpu().detach().numpy()

with open('valid_est.npy', 'wb') as file:
    np.save(file, y_pred_valid)

y_pred_test = np.ndarray(shape=(100, 2, 64, 64))
for idx in range(int(100 / batch_size)):
    x_test_ = x_test[idx * batch_size: (idx + 1) * batch_size, :, :, :]

    x_test_batch = x_test_.cuda()
    y_pred_test_ = net(x_test_batch)

    y_pred_test[idx * batch_size: (idx + 1) * batch_size, :, :, :] = y_pred_test_.cpu().detach().numpy()

with open('test_est.npy', 'wb') as file:
    np.save(file, y_pred_test)

plt.plot(np.arange(epochs), np.array(loss_train_arr), label='train')
plt.plot(np.arange(epochs), np.array(loss_valid_arr), label='validation')
plt.title('Model Loss')
plt.legend(loc='upper right')
plt.savefig('../loss.png')
plt.show()