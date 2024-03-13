# loading file for pretrained models

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import backbones_unet
from backbones_unet.model.unet import Unet
from backbones_unet.model.losses import DiceLoss,focal_loss,FocalLoss
from backbones_unet.utils.trainer import Trainer


def load(config):

    model = Unet(
    	backbone = config.backbone,
    	in_channels = config.input_channels,
    	num_classes = 2
    	)

    return model, config.backbone

