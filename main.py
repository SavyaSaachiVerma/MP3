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

"""Directory where the the data sets will be stored"""
data_dir = "./Data"

"""HYPER-PARAMETERS"""

"""number of input samples in one batch for training and testing"""
batch_size_train = 64
batch_size_test = 32
"""learning rate for optimization algorithm (USED ADAM)"""
learning_rate = 0.001
"""number of times to traverse the entire training dataset"""
epochs = 1
"""bool: specifies if the model's state should be loaded from the checkpoint file"""
load_chkpt = False


def main():

    """Transformations for Augmenting + Conversion to tensors fot PIL Images in Training and Test Dataset
    Referenced: Stanford CS231n: Convolutional Neural Networks for Visual Recognition, Lecture notes, "Convolutional
    Neural Networks (CNNs / ConvNets)"""

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

    """Setting seed values manually"""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    """Downloading the data sets by applying the above transformations from torchvision dataset library"""
    train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
                                            transform=augment_train_ds)
    test_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                           transform=augment_test_ds)

    """Using the Dataloader to divide the data into batches and performing shuffling """
    train_ds_loader = data.DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=4)
    test_ds_loader = data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=False, num_workers=4)

    """Setting device to CUDA if it is available"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Initializing Model")
    conv_net = ConvNeuralNet()
    conv_net = conv_net.to(device)
    start_epoch = 0

    if load_chkpt:
        print("Saved Model is being loaded")
        chkpt = torch.load('./Checkpoint/model_state.pt')
        conv_net.load_state_dict(chkpt['conv_net_model'])
        start_epoch = chkpt['epoch']

    """If multiple GPUs are available then use asynchronous training """
    if device == 'cuda':
        conv_net = torch.nn.DataParallel(conv_net)
        cudnn.benchmark = True

    """___________ Training ___________"""

    print("Starting Training")

    """Criterion Function: Softmax + Log-Likelihood"""
    loss_fn = nn.CrossEntropyLoss()
    """Adam Optimizer (as it takes advantage of both RMSDrop and Momentum"""
    optimizer = optim.Adam(conv_net.parameters(), lr=learning_rate)

    for epoch in range(start_epoch, epochs):

        cur_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i, (inputs, labels) in enumerate(train_ds_loader):

            """Transfer inputs and labels to CUDA if available"""
            inputs = inputs.to(device)
            labels = labels.to(device)

            """Loss function requires the inputs to be wrapped in variables"""
            inputs = Variable(inputs)

            """Torch tends to take cumulative gradients which is not required so setting it to zero after each batch"""
            optimizer.zero_grad()

            outputs = conv_net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()

            """ Overflow error in the optimizer if the step size is not reset."""
            if epoch > 8:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if state['step'] >= 1024:
                            state['step'] = 1000
            optimizer.step()

            cur_loss += loss.item()
            cur_loss /= (i + 1)

            _, predicted_label = torch.max(outputs, 1)
            # print(predicted_label.shape, labels.shape)
            total_samples += labels.shape[0]
            # arr = (predicted_label == labels).numpy()
            # print(np.sum(arr))
            """can not use numpy as the tensors are in CUDA"""
            total_correct += predicted_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples

            if i % 100 == 0:
                print('Training [epoch: %d, batch: %d] loss: %.3f, accuracy: %.5f' %
                      (epoch + 1, i + 1, cur_loss, accuracy))

        """Saving model after every 5 epochs"""
        if (epoch + 1) % 5 == 0:
            print('==> Saving model ...')
            state = {
                'conv_net_model': conv_net.state_dict(),
                'epoch': epoch,
            }
            if not os.path.isdir('./Checkpoint'):
                os.mkdir('Checkpoint')
            torch.save(state, './Checkpoint/model_state.pt')

    print("Training Completed!")

    """___________ Testing ____________"""
    print("Testing Started")
    """Puts model in testing state"""
    conv_net.eval()

    cur_loss = 0
    total_correct = 0
    total_samples = 0
    """Do testing under the no_grad() context so that torch does not store/use these actions to calculate gradients"""
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_ds_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = Variable(inputs)

            outputs = conv_net(inputs)
            loss = loss_fn(outputs, labels)

            cur_loss += loss.item()
            cur_loss /= (i + 1)

            _, predicted_label = torch.max(outputs, 1)
            total_samples += labels.shape[0]
            # arr = (predicted_label == labels).numpy()
            total_correct += predicted_label.eq(labels.long()).float().sum().item()
            accuracy = total_correct / total_samples

            if i % 50 == 0:
                print('Testing [batch: %d] loss: %.3f, accuracy: %.5f' %
                      (i + 1, cur_loss, accuracy))

    print("Testing Completed with accuracy:" + str(accuracy))


if __name__ == "__main__":
    main()
