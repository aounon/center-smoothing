import torch
import torch.nn as nn
import torch.nn.functional as F

from facenet_pytorch import MTCNN
from pytorch_pretrained_biggan import BigGAN, one_hot_from_names

from random import randrange

from tools import repeat_along_dim


class GANpretrained(nn.Module):
    def __init__(self, class_name: str):
        super(GANpretrained, self).__init__()
        self.model = BigGAN.from_pretrained('biggan-deep-128').to('cuda')
        self.class_vector = torch.squeeze(torch.from_numpy(one_hot_from_names([class_name])), dim=0).to('cuda')

    def forward(self, x):
        num_vec = x.shape[0]
        cls_vec = repeat_along_dim(self.class_vector, num_vec)
        return self.model(x, cls_vec, 0.5)

class FaceDetector(nn.Module):
    def __init__(self, device):
        super(FaceDetector, self).__init__()
        self.device = device
        self.mtcnn = MTCNN(device=self.device)

    def boxes_tensor(self, boxes):
        num_boxes = boxes.shape[0]
        boxes_tensor = torch.full((num_boxes, 4), -1)

        for i in range(num_boxes):
            boxes_list = boxes[i]
            if boxes_list is not None:
                boxes_tensor[i] = torch.tensor(boxes_list[0])

        boxes_tensor = boxes_tensor.cuda(device=self.device)
        return boxes_tensor

    def forward(self, x):
        boxes, _ = self.mtcnn.detect(x)
        return self.boxes_tensor(boxes)


class Reconstructor(nn.Module):
    def __init__(self):
        super(Reconstructor, self).__init__()

        # Input is expected in the shape: batch_size * 3 * 32 * 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # 32 * 30 * 30
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2)  # 64 * 28 * 28
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)  # 128 * 26 * 26
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.bn3(x)
        x = torch.sigmoid(self.deconv3(x))
        return x

class AutoencoderMNIST(nn.Module):
    def __init__(self, code_size: int = 128):
        super(AutoencoderMNIST, self).__init__()

        # encoder
        # Input is expected in the shape: batch_size * 1 * 28 * 28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # 32 * 28 * 28
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)  # 64 * 14 * 14
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  # 128 * 7 * 7
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 7 * 7, code_size)

        # decoder
        self.fc2 = nn.Linear(code_size, 128 * 7 * 7)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))  # Low-dimensional code
        return x

    def decode(self, x):
        x = F.relu(self.fc2(x))
        x_shape = x.size()
        x = x.view((x_shape[0], 128, 7, 7))
        x = self.bn3(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.bn4(x)
        x = torch.sigmoid(self.deconv3(x))
        return x

    def forward(self, x):
        # Encode
        x = self.encode(x)

        # Decode
        x = self.decode(x)
        return x


class AutoencoderCIFAR(nn.Module):
    def __init__(self, code_size: int = 256):
        super(AutoencoderCIFAR, self).__init__()

        # encoder
        # Input is expected in the shape: batch_size * 3 * 32 * 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # 32 * 32 * 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)  # 64 * 16 * 16
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  # 128 * 8 * 8
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, code_size)

        # decoder
        self.fc2 = nn.Linear(code_size, 128 * 8 * 8)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))  # Low-dimensional code
        return x

    def decode(self, x):
        x = F.relu(self.fc2(x))
        x_shape = x.size()
        x = x.view((x_shape[0], 128, 8, 8))
        x = self.bn3(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.bn4(x)
        x = torch.sigmoid(self.deconv3(x))
        return x

    def forward(self, x):
        # Encode
        x = self.encode(x)

        # Decode
        x = self.decode(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # encoder
        # Input is expected in the shape: batch_size * 3 * 32 * 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)     # 32 * 30 * 30
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)    # 64 * 14 * 14
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2)   # 128 * 6 * 6
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)

        # decoder
        self.fc2 = nn.Linear(256, 128 * 6 * 6)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1)

    def forward(self, x):
        # Encode
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x_shape = x.size()
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))         # Low-dimensional code

        # Decode
        x = F.relu(self.fc2(x))
        x = x.view(x_shape)
        x = self.bn3(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.bn4(x)
        x = torch.sigmoid(self.deconv3(x))
        return x


class ReconstructorMNIST(nn.Module):
    def __init__(self, A_mat):
        super(ReconstructorMNIST, self).__init__()

        self.A_mat = A_mat

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # 32 * 28 * 28
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)  # 64 * 14 * 14
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  # 128 * 7 * 7
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        A_t = torch.transpose(self.A_mat, 0, 1)
        x = torch.matmul(x, A_t)
        x = x.view((x.shape[0], 1, 28, 28))

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.bn3(x)
        x = torch.sigmoid(self.deconv3(x))
        return x


class ReconstructorCIFAR(nn.Module):
    def __init__(self, A_mat):
        super(ReconstructorCIFAR, self).__init__()

        self.A_mat = A_mat

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # 32 * 32 * 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)  # 64 * 16 * 16
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  # 128 * 8 * 8
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        A_t = torch.transpose(self.A_mat, 0, 1)
        x = torch.matmul(x, A_t)
        x = x.view((x.shape[0], 3, 32, 32))

        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.bn3(x)
        x = torch.sigmoid(self.deconv3(x))
        return x


class FaceRecCelebA(nn.Module):
    def __init__(self):
        super(FaceRecCelebA, self).__init__()

        # Input is expected in the shape: batch_size * 3 * 64 * 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1)  # 32 * 32 * 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)  # 128 * 16 * 16
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)  # 128 * 8 * 8
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x


class FaceRecCelebAold(nn.Module):
    def __init__(self):
        super(FaceRecCelebAold, self).__init__()

        # Input is expected in the shape: batch_size * 3 * 218 * 178
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=0)  # 32 * 108 * 88
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)  # 64 * 54 * 44
        self.bn1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)  # 128 * 27 * 22
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 4), stride=(2, 2), padding=(0, 1))  # 128 * 12 * 11
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 12 * 11, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x