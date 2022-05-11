import torch.nn as nn


def get_simple_model(hidden_channels=10, kernel_size=7, out_channels=1):
    assert kernel_size % 2 == 1, "Kernel size should be odd"
    padding = int(kernel_size / 2 - .5)

    return nn.Sequential(
        nn.Conv2d(1, hidden_channels, kernel_size, padding=padding),
        nn.ReLU(),
        nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding),
        nn.ReLU(),
        nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=padding),
        nn.Sigmoid()
    )
