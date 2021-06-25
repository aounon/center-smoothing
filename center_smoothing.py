import torch
import torch.nn as nn

import numpy as np
from math import ceil, sqrt, log
from scipy.stats import norm

from tools import repeat_along_dim

class Smooth(object):

    def __init__(self, base_function: nn.Module, dist_fn, sigma: float, n_pred: int = 10 ** 4, n_cert: int = 10 ** 6,
                 n_cntr: int = 30, alpha_1: float = 0.005, alpha_2: float = 0.005, delta = 0.05, radius_coeff = 3,
                 output_is_hd: bool = False):
        self.base_function = base_function
        self.dist_fn = dist_fn
        self.sigma = sigma
        self.n_pred = n_pred    # number of samples used for prediction
        self.n_cert = n_cert    # number of samples used for certification
        self.n_cntr = n_cntr    # number of candidate centers
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.delta = delta
        self.radius_coeff = radius_coeff    # for the relaxed triangle inequality
        self.output_is_hd = output_is_hd    # whether to use procedure for high-dimensional outputs

    def certify(self, input: torch.tensor, eps_in: float, batch_size: int = 1000):
        with torch.no_grad():
            # Computing center
            if self.output_is_hd:
                center, is_good = self.compute_center_hd(input, batch_size=batch_size)
            else:
                center, is_good = self.compute_center(input, batch_size=batch_size)

            if not is_good:
                return -1.0, -1.0

            # Calculating smoothing error
            model_output = torch.squeeze(self.base_function(repeat_along_dim(input, 1)), 0)
            smoothing_error = self.dist_fn(repeat_along_dim(center, 1), repeat_along_dim(model_output, 1))

            # Computing certificate
            min_prob = 0.5 + self.delta
            quantile = norm.cdf(norm.ppf(min_prob) + (eps_in / self.sigma)) + sqrt(log(1 / self.alpha_2)/(2 * self.n_cert))

            if quantile > 1.0 or quantile < 0.0:
                print('Invalid quantile value: %.3f' % quantile)
                return -1.0, smoothing_error

            dist = np.zeros(self.n_cert)
            num = self.n_cert

            batch_inp = repeat_along_dim(input, batch_size)
            batch_cen = repeat_along_dim(center, batch_size)

            for i in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                if this_batch_size != batch_size:
                    batch_inp = repeat_along_dim(input, this_batch_size)
                    batch_cen = repeat_along_dim(center, this_batch_size)

                noise = torch.randn_like(batch_inp) * self.sigma
                samples = self.base_function(batch_inp + noise)

                dist_batch = self.dist_fn(samples, batch_cen)
                start = i * batch_size
                end = start + this_batch_size
                dist[start : end] = dist_batch

            eps_out = self.radius_coeff * np.quantile(dist, quantile)   # 3 * np.quantile(dist, quantile)

            return eps_out, smoothing_error


    def compute_center(self, input: torch.tensor, batch_size: int = 1000):
        # Smoothing procedure
        with torch.no_grad():
            delta_1 = sqrt(log(2 / self.alpha_1) / (2 * self.n_pred))
            is_good = False
            num = self.n_pred

            inp_batch = repeat_along_dim(input, batch_size)

            for i in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                if this_batch_size != batch_size:
                    inp_batch = repeat_along_dim(input, this_batch_size)

                noise = torch.randn_like(inp_batch) * self.sigma
                if i == 0:
                    samples = self.base_function(inp_batch + noise)
                else:
                    samples = torch.cat((samples, self.base_function(inp_batch + noise)), 0)

            center, radius = self.meb(samples)
            num_pts = self.pts_in_nbd(input, center, radius, batch_size=batch_size)

            frac = num_pts / self.n_pred
            p_delta_1 = frac - delta_1
            delta_2 = (1 / 2) - p_delta_1

            print(max(delta_1, delta_2))
            if max(delta_1, delta_2) <= self.delta:
                is_good = True
            else:
                print('Bad center. Abstaining ...')

        return center, is_good

    def compute_center_hd(self, input: torch.tensor, batch_size: int = 1000):
        # Smoothing procedure for high-dimensional outputs
        with torch.no_grad():
            inp_batch = repeat_along_dim(input, self.n_cntr)
            noise = torch.randn_like(inp_batch) * self.sigma
            candidate_centers = self.base_function(inp_batch + noise)

            dist = np.zeros((self.n_cntr, self.n_pred))

            delta_1 = sqrt(log(2 / self.alpha_1) / (2 * self.n_pred))
            is_good = False
            num = self.n_pred

            inp_batch = repeat_along_dim(input, batch_size)

            for i in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                if this_batch_size != batch_size:
                    inp_batch = repeat_along_dim(input, this_batch_size)

                noise = torch.randn_like(inp_batch) * self.sigma
                samples = self.base_function(inp_batch + noise)

                for cntr_idx in range(self.n_cntr):
                    batch_cen = repeat_along_dim(candidate_centers[cntr_idx], this_batch_size)
                    dist_batch = self.dist_fn(samples, batch_cen)

                    start = i * batch_size
                    end = start + this_batch_size
                    dist[cntr_idx, start : end] = dist_batch

            median_dist = np.median(dist, axis=1)
            min_idx = np.argmin(median_dist)
            center = candidate_centers[min_idx]
            radius = median_dist[min_idx]

            num_pts = self.pts_in_nbd(input, center, radius, batch_size=batch_size)

            frac = num_pts / self.n_pred
            p_delta_1 = frac - delta_1
            delta_2 = (1 / 2) - p_delta_1

            if max(delta_1, delta_2) <= self.delta:
                is_good = True
            else:
                print('Bad center. Abstaining ...')

        return center, is_good


    def pts_in_nbd(self, input: torch.tensor, center: torch.tensor, radius, batch_size: int = 1000):
        with torch.no_grad():
            num = self.n_pred
            num_pts = 0

            batch_inp = repeat_along_dim(input, batch_size)
            batch_cen = repeat_along_dim(center, batch_size)

            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                if this_batch_size != batch_size:
                    batch_inp = repeat_along_dim(input, this_batch_size)
                    batch_cen = repeat_along_dim(center, this_batch_size)

                noise = torch.randn_like(batch_inp) * self.sigma
                samples = self.base_function(batch_inp + noise)

                dist = self.dist_fn(samples, batch_cen)
                num_pts += np.sum(np.where(dist <= radius, 1, 0))
        return num_pts

    def meb(self, samples):
        with torch.no_grad():
            radius = np.inf
            num_samples = samples.shape[0]
            for i in range(num_samples):
                curr_sample = samples[i]
                sample_batch = repeat_along_dim(curr_sample, num_samples)
                dist = self.dist_fn(samples, sample_batch)
                median_dist = np.median(dist)
                if median_dist < radius:
                    radius = median_dist
                    center = curr_sample

        return center, radius
