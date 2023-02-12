import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from django.utils.baseconv import base64
from rest_framework.response import Response
from torch.utils.data import DataLoader
from rest_framework.views import APIView
import base64
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
#
# from core.models import File
# from core.serializers import FileSerializer


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(channels_noise, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 128
FEATURES_GEN = 64



GenTrained = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
GenTrained.load_state_dict(torch.load('/home/harsh/Desktop/my_proj/GANdeplot/DeployGAN/staticfiles/Gen4.pth', map_location='cpu'))

image_path = "/home/harsh/Desktop/my_proj/GANdeplot/DeployGAN/staticfiles/Saved.jpg"


def getImage():
    noise = torch.randn(1, Z_DIM, 1, 1).to(device)
    GenTrained.eval()
    DiffTrained.eval()
    fake = GenTrained(noise)
    img_grid_fake = torchvision.utils.make_grid(
        fake, normalize=True
    )
    torchvision.utils.save_image(img_grid_fake, image_path)


class FileView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    def get(self, request, *args, **kwargs):
        getImage()
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        return Response(image_data)


