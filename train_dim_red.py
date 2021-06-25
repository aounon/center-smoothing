import argparse

import torch
import torchvision
from torchvision import transforms

import torch.optim as optim
import torch.nn as nn

from models import AutoencoderCIFAR, AutoencoderMNIST

from math import sqrt

import sys


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=0.0)

    args = parser.parse_args()

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transforms.ToTensor())
        model = AutoencoderCIFAR(args.latent_dim).cuda()

    elif args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transforms.ToTensor())
        model = AutoencoderMNIST(args.latent_dim).cuda()

    else:
        sys.exit('Unrecognized dataset name')

    batch_size = 1000
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    criterion = nn.MSELoss().cuda() # nn.BCELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if torch.cuda.is_available():
        print('GPU: %s' % torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('GPU not available. Current device: %s' % torch.cuda.get_device_name(torch.cuda.current_device()))

    log = open(args.model_path + '.csv', "w")
    log.write('Epoch\tLoss\n')
    log.flush()

    for epoch in range(100):  # loop over the dataset multiple times

        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, _ = data
            inputs = inputs.cuda()

            # Adding Gaussian noise
            inputs_noisy = inputs + (torch.randn_like(inputs) * args.sigma)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs_noisy)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

        # Print and log loss for this epoch
        print('Epoch: %d, loss: %.3f' % (epoch, loss.item()))
        log.write('%d\t%.3f\n' % (epoch, loss.item()))
        log.flush()

        # Save model parameters after every ten epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), args.model_path)

    print('Finished Training')

    torch.save(model.state_dict(), args.model_path)
    log.close()


if __name__ == "__main__":
    main()
