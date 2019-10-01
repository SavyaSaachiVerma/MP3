import os
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils import data
from ConvNeuralNetModel import ConvNeuralNet

data_dir = "./Data"
batch_size_train = 64
batch_size_test = 32
learning_rate = 0.001
epochs = 5
load_chkpt = False

def main():

    """Transformations for Augmenting Training and Test Dataset"""
    augment_train_ds = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ])

    augment_test_ds = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
    ])

    """Set seed."""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


    """Loading the datasets from torchvision dataset library"""
    train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
                                            transform=augment_train_ds)
    test_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                           transform=augment_test_ds)
    train_ds_loader = data.DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=4)
    test_ds_loader = data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=False, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Initializing Model")
    conv_net = ConvNeuralNet()
    conv_net = conv_net.to(device)
    start_epoch = 0

    if load_chkpt:
        print("Loading Saved Model")
        chkpt = torch.load('./Checkpoint/model_state.pt')
        conv_net.load_state_dict(chkpt['conv_net_model'])
        start_epoch = chkpt['epoch']

    if device == 'cuda':
        conv_net = torch.nn.DataParallel(conv_net)
        cudnn.benchmark = True

    """Training"""
    print("Starting Training")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(conv_net.parameters(), lr=learning_rate)

    for epoch in range(start_epoch, epochs):

        cur_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i, (inputs, labels) in enumerate(train_ds_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = Variable(inputs)

            optimizer.zero_grad()

            outputs = conv_net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()

            if epoch > 16:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if state['step'] >= 1024:
                            state['step'] = 1000
            optimizer.step()

            cur_loss += loss.item()
            cur_loss /= (i+1)

            _, predicted_label = torch.max(outputs, 1)
            # print(predicted_label.shape, labels.shape)
            total_samples += labels.shape[0]
            arr = (predicted_label == labels).numpy()
            # print(np.sum(arr))
            total_correct += np.sum(arr)
            accuracy = total_correct/total_samples

            print('Training [epoch: %d, batch: %d] loss: %.3f, accuracy: %.5f' %
                  (epoch + 1, i + 1, cur_loss, accuracy))

        if epoch % 5 == 0:
            print('==> Saving model ...')
            state = {
                'conv_net_model': conv_net.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir('./Checkpoint'):
                os.mkdir('Checkpoint')
            torch.save(state, './Checkpoint/model_state.pt')

    print("Training Completed")

    """Testing"""
    print("Testing Started")
    conv_net.eval()

    cur_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_ds_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = Variable(inputs)

            outputs = conv_net(inputs)
            loss = loss_fn(outputs, labels.long)

            cur_loss += loss.item()
            cur_loss /= (i+1)

            _, predicted_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            arr = (predicted_label == labels).numpy()
            total_correct += np.sum(arr)
            accuracy = total_correct / total_samples

            print('Testing [batch: %d] loss: %.3f, accuracy: %.5f' %
                  (i + 1, cur_loss, accuracy))
    print("Testing Completed with accuracy:" + accuracy)


if __name__ == "__main__":
    main()

