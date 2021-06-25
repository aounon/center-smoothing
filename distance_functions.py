import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
import numpy as np
import math

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to('cuda') # closer to "traditional" perceptual loss, when used for optimization


def normalize_in_range(batch, range_max=1, range_min=-1):
    batch_flat = torch.flatten(batch, start_dim=1)
    batch_min = torch.min(batch_flat, dim=1)[0]
    batch_min = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(batch_min, dim=1), dim=1), dim=1)

    batch = batch - batch_min

    batch_flat = torch.flatten(batch, start_dim=1)
    batch_max = torch.max(batch_flat, dim=1)[0]
    batch_max = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(batch_max, dim=1), dim=1), dim=1)

    batch = batch / batch_max

    batch = (batch * (range_max - range_min)) + range_min
    return batch


def perceptual_dist(batch1, batch2):
    batch1 = normalize_in_range(batch1)
    batch2 = normalize_in_range(batch2)
    dist = loss_fn_vgg(batch1, batch2)
    dist = torch.squeeze(dist)
    dist = dist.cpu().numpy()
    return dist


def angular_distance(batch1, batch2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    b1_flat = torch.flatten(batch1, start_dim=1)
    b2_flat = torch.flatten(batch2, start_dim=1)
    sim = cos(b1_flat, b2_flat)
    dist = torch.acos(sim) / math.pi
    dist = dist.cpu().numpy()
    return dist


def total_variation(batch: torch.tensor, p: int = 1):
    # batch: torch tensor containing a batch of images idexed as batch_size * color_channel * height * width
    diff1 = torch.flatten(batch[:, :, 1:, :] - batch[:, :, :-1, :], start_dim=2)
    diff2 = torch.flatten(batch[:, :, :, 1:] - batch[:, :, :, :-1], start_dim=2)
    diff = torch.cat((diff1, diff2), dim=2)
    dist = torch.sum(torch.norm(diff, p=p, dim=1), dim=1)
    dist = dist.cpu().numpy()
    return dist


def tv1_diff(batch1: torch.tensor, batch2: torch.tensor):
    return total_variation(batch1 - batch2, 1)

def tv2_diff(batch1: torch.tensor, batch2: torch.tensor):
    return total_variation(batch1 - batch2, 2)

####### Jaccard Distance ########

def rect_intersection(r1, r2):
    r3 = torch.zeros_like(r1)
    r3[:, [0, 1]] = torch.max(r1[:, [0, 1]], r2[:, [0, 1]])
    r3[:, [2, 3]] = torch.min(r1[:, [2, 3]], r2[:, [2, 3]])
    r3[(r3[:, 0] > r3[:, 2]) | (r3[:, 1] > r3[:, 3]), :] = -1.0
    return r3


def rect_area(r):
    return (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])


def iou(r1, r2):
    intersection = rect_intersection(r1, r2)
    area_intersection = rect_area(intersection)
    area_union = rect_area(r1) + rect_area(r2) - area_intersection

    return area_intersection/area_union


def jaccard_dist(batch1, batch2):
    dist = 1 - iou(batch1, batch2)
    dist = dist.cpu().numpy()
    return dist

######## L2 distance ##########

def l2_dist(batch1, batch2):
    dist = torch.norm(torch.flatten(batch1 - batch2, start_dim=1), p=2, dim=1)
    dist = dist.cpu().numpy()
    return dist
