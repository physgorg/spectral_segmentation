# pretrained convNet 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from backbones_unet.model.unet import Unet
from backbones_unet.model.losses import DiceLoss,focal_loss,FocalLoss
from backbones_unet.utils.trainer import Trainer

from forge import flags

flags.DEFINE_string(
    "backbone",
    'convnext_pico_ols',
    "Pretrained backbone to use."
)
flags.DEFINE_integer(
    "n_classes",
    2,
    "Number of output classes."
)



def load(config):

    model = Unet(
    	backbone = config.backbone,
    	in_channels = config.input_size,
    	num_classes = config.n_classes)

    return model, config.backbone

