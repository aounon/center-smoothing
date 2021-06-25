import argparse

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from models import AutoencoderCIFAR, AutoencoderMNIST
from tools import imgsave
from center_smoothing import Smooth

from distance_functions import tv1_diff, tv2_diff, l2_dist, angular_distance

import time


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("distance_function", type=str)
    parser.add_argument("logfile", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("--eps_in", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--n_pred", type=int, default=10 ** 4)
    parser.add_argument("--n_cert", type=int, default=10 ** 6)

    args = parser.parse_args()

    if args.dataset == 'cifar10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transforms.ToTensor())
        model = AutoencoderCIFAR(args.latent_dim).cuda()

    elif args.dataset == 'mnist':
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transforms.ToTensor())
        model = AutoencoderMNIST(args.latent_dim).cuda()

    else:
        sys.exit('Unrecognized dataset name')

    if args.distance_function == 'tv1':
        dist_fn = tv1_diff
    elif args.distance_function == 'tv2':
        dist_fn = tv2_diff
    elif args.distance_function == 'angular_distance':
        dist_fn = angular_distance
    elif args.distance_function == 'L2':
        dist_fn = l2_dist
    else:
        sys.exit('Unrecognized distance function')

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=True, num_workers=2)

    dataiter = iter(testloader)
    logger = open(args.logfile, "w")
    logger.write('eps_in = %.3f\tsigma = %.3f\niter\teps_out\tsmoothing_error\ttime\n' % (args.eps_in, args.sigma))
    logger.flush()

    model.load_state_dict(torch.load(args.model_path))
    model_smooth = Smooth(model, dist_fn, args.sigma, n_pred=args.n_pred, n_cert=args.n_cert)

    for i in range(50):
        images, _ = dataiter.next()
        images = images.cuda()
        input = images[0]

        start = time.time()
        eps_out, smoothing_error = model_smooth.certify(input, args.eps_in)
        end = time.time()
        time_diff = end - start

        print('iter = %d\teps_out = %.3f\tsmoothing_error = %.3f\ttime = %.3f' % (i, eps_out, smoothing_error, time_diff))
        logger.write('%d\t%.3f\t%.3f\t%.3f\n' % (i, eps_out, smoothing_error, time_diff))
        logger.flush()

    logger.close()


if __name__ == "__main__":
    main()