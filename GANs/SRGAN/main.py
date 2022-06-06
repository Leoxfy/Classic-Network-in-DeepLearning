import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    # Conv BN Leaky/PReLU
    def __init__(
            self,
            in_channels,
            out_channels,
            discriminator=False,
            use_act=True,
            use_bn=True,
            **kwargs
    ):
        super(ConvBlock, self).__init__()
        self.cnn = nn.Conv2d



class UpsampleBlock(nn.Module):
    pass


class ResidualBlock(nn.Module):
    pass


class Generator(nn.Module):
    pass


class Discriminator(nn.Module):
    pass