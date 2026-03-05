import torch.nn as nn
import torch
import numpy as np
from torchvision.models import efficientnet_b3

class global_partial_crossattention_model(nn.Module):
    def __init__(self, 
                 ):
        super().__init__()

        self.local_encoder=efficientnet_b3(weights="DEFAULT")
        self.global_encoder=nn.Sequential(
            nn.Conv2d(3,64,(8,8),(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((8,8)),
            nn.Conv2d(64,256,),
        )