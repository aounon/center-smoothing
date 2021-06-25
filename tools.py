import torch
from torch.utils.data import Subset
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

def repeat_along_dim(input: torch.tensor, num_repeat: int, dim: int = 0):
    return torch.repeat_interleave(torch.unsqueeze(input, dim), num_repeat, dim=dim)


def project(inputs, A, sigma=0.0):
    inp_shape = inputs.shape
    inputs = torch.flatten(inputs, start_dim = 1)
    y = torch.matmul(inputs, A)
    y_noisy = y + (torch.randn_like(y) * sigma)
    A_t = torch.transpose(A, 0, 1)
    inputs = torch.matmul(y_noisy, A_t)
    inputs = inputs.view(inp_shape)
    return inputs

def create_subdataset(dataset, class_name: str):
    class_indices = []
    class_idx = dataset.class_to_idx[class_name]

    for i in range(len(dataset)):
        current_class = dataset[i][1]
        if current_class == class_idx:
            class_indices.append(i)

    return Subset(dataset, class_indices)

# functions to show an image
def imgsave(img, filename):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(filename) # plt.draw()
    plt.close()


def plot_landmarks(img, landmarks, filename):
    npimg = img.numpy()
    nplandmarks = landmarks.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    for i in range(5):
        x, y = nplandmarks[2*i], nplandmarks[(2*i)+1]

        plt.scatter(x, y, c = 'g')

    plt.savefig(filename) # plt.draw()
    plt.close()


def plot_bbox(img, bbox, filename):
    npimg = img.numpy()
    npbbox = bbox.numpy()
    print(npbbox)

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.scatter(npbbox[0], npbbox[1], c = 'g')
    plt.scatter(npbbox[0] + npbbox[2], npbbox[1] + npbbox[3], c='g')

    plt.savefig(filename) # plt.draw()
    plt.close()

celeba_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()])
