import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3
import random


class LaplacianConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.register_buffer("weight", kernel)

    def forward(self, x):
        if x.max() <= 1.0:
            x = x * 255.0
        b, c, h, w = x.shape
        x_reshape = x.view(b * c, 1, h, w)
        out = F.conv2d(x_reshape, self.weight, padding=1)
        out = out.view(b, c, h, w)
        return torch.clamp(out, min=-3.0, max=3.0)


class NoiseExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.laplacian = LaplacianConv2d()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.net(self.laplacian(x))


class DiscriminatorModel(nn.Module):
    def __init__(self, hidden_dim=512, noise_weight=3.0):
        super().__init__()
        self.noise_weight = noise_weight

        self.rgb_encoder = efficientnet_b3(weights="DEFAULT")
        self.rgb_encoder.classifier = nn.Linear(1536, 256)

        self.noise_encoder = NoiseExtractor()
        self.noise_projector = nn.Sequential(
            nn.Linear(128, 256), nn.LayerNorm(256), nn.ReLU()
        )

        self.feature_dim = 256 + 256

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward_one_branch(self, img, force_drop_rgb=False):
        if self.training and force_drop_rgb:
            f_rgb = torch.zeros((img.size(0), 256), device=img.device)
        else:
            f_rgb = self.rgb_encoder(img)

        raw_noise = self.noise_encoder(img)
        f_noise = self.noise_projector(raw_noise)
        f_noise = f_noise * self.noise_weight

        return torch.cat([f_rgb, f_noise], dim=1)

    def forward(self, img_center, img_cand):
        drop_rgb = False
        if self.training and random.random() < 0.2:
            drop_rgb = True

        v_center = self.forward_one_branch(img_center, force_drop_rgb=drop_rgb)
        v_cand = self.forward_one_branch(img_cand, force_drop_rgb=drop_rgb)

        diff = torch.abs(v_center - v_cand)
        prod = v_center * v_cand
        combined = torch.cat([v_center, v_cand, diff, prod], dim=1)

        return self.classifier(combined)
