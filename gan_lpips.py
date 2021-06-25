import argparse
import time

import torch
from pytorch_pretrained_biggan import truncated_noise_sample

from PIL import Image
from models import GANpretrained

from center_smoothing import Smooth
from distance_functions import perceptual_dist

import nltk
nltk.download('wordnet')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", type=str)
    parser.add_argument("--eps_in", type=float)
    parser.add_argument("--sigma", type=float)

    args = parser.parse_args()

    logger = open(args.logfile, "w")
    logger.write('eps_in = %.3f\tsigma = %.3f\niter\tsmoothing_error\teps_out\ttime\n' % (
        args.eps_in, args.sigma))
    logger.flush()

    # Load pre-trained model tokenizer (vocabulary)
    model = GANpretrained('Butterfly')
    model_smooth = Smooth(model, perceptual_dist, args.sigma, n_pred=2000, n_cert=10000, delta=0.08,
                          radius_coeff=10, output_is_hd=True)

    for i in range(50):
        # Prepare an input
        truncation = 0.5
        latent_vector = truncated_noise_sample(truncation=truncation, batch_size=1)
        latent_vector = torch.from_numpy(latent_vector).to('cuda')
        latent_vector = torch.squeeze(latent_vector, dim=0)

        start = time.time()
        eps_out, smoothing_error = model_smooth.certify(latent_vector, args.eps_in, batch_size=150)
        end = time.time()
        time_diff = end - start

        print('iter = %d\tsmoothing_error = %.3f\teps_out = %.3f\ttime = %.3f' % (
            i, smoothing_error, eps_out, time_diff))
        logger.write('%d\t%.3f\t%.3f\t%.3f\n' % (i, smoothing_error, eps_out, time_diff))
        logger.flush()

    logger.close()


if __name__ == "__main__":
    main()
