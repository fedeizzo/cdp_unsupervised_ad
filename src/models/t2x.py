import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def make_res_layer(in_channels, hidden_channels, kernel_size):
    kernel = (kernel_size, kernel_size)
    padding = ((kernel_size - 1) // 2, (kernel_size - 1) // 2)
    return Residual(
        nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel, padding=padding),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel, padding=padding),
            nn.ReLU()
        )
    )


def get_t2x_model():
    return nn.Sequential(
        nn.Conv2d(1, 3, (5, 5), padding=(2, 2)),
        nn.ReLU(),
        make_res_layer(3, 10, 5),
        make_res_layer(3, 10, 3),
        nn.Conv2d(3, 1, (3, 3), padding=(1, 1)),
        nn.Conv2d(1, 1, (3, 3), padding=(1, 1))
    )
