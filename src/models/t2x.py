import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def get_t2x_model():
    from models.pix2pix import UnetGenerator
    return UnetGenerator(1, 1, 9)
