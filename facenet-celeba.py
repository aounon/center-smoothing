import argparse

import torch
import numpy as np
from PIL import Image

from center_smoothing import Smooth

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from random import randrange
from iou_torch import jaccard_dist

from models import FaceDetector
import os.path

import time


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("celeba_path", type=str)
    parser.add_argument("logfile", type=str)
    parser.add_argument("--eps_in", type=float)
    parser.add_argument("--sigma", type=float)

    args = parser.parse_args()

    logger = open(args.logfile, "w")
    logger.write('eps_in = %.3f\tsigma = %.3f\niter\tsmoothing_error\teps_out\ttime\n' % (
    args.eps_in, args.sigma))
    logger.flush()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = FaceDetector(device)
    model_smooth = Smooth(model, jaccard_dist, args.sigma * 255, n_pred=5000, n_cert=10000)

    for i in range(50):
        img_id = randrange(202600) + 10 ** 6
        filepath = os.path.join(args.celeba_path, str(img_id)[1:] + ".jpg")

        img = Image.open(filepath)
        img = np.asarray(img)
        input = torch.tensor(img, dtype=float)
        input = input.cuda(device=device)

        start = time.time()
        eps_out, smoothing_error = model_smooth.certify(input, args.eps_in)
        end = time.time()
        time_diff = end - start

        print('iter = %d\tsmoothing_error = %.3f\teps_out = %.3f\ttime = %.3f' % (
        i, smoothing_error, eps_out, time_diff))
        logger.write('%d\t%.3f\t%.3f\t%.3f\n' % (i, smoothing_error, eps_out, time_diff))
        logger.flush()

    logger.close()


if __name__ == "__main__":
    main()
