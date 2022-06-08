import torch.nn as nn
from utils.utils import Mode


class ResidualBlock(nn.Module):
    """
    Residual Block with instance normalization
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return x + self.residual_block(x)


def _get_bottleneck_model(mode, conv_dim, repeat_num):
    out_channels = 1 if mode in [Mode.MODE_T2X, Mode.MODE_X2T, Mode.MODE_BOTH] else 2

    layers = [nn.Conv2d(1, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
              nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True), nn.ReLU(inplace=True)]

    # Down-sampling
    curr_dim = conv_dim
    for _ in range(2):
        layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2

    # Bottleneck
    for _ in range(repeat_num):
        layers.append(ResidualBlock(in_channels=curr_dim, out_channels=curr_dim))

    # Up-sampling
    for _ in range(2):
        layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim // 2

    layers.append(nn.Conv2d(curr_dim, out_channels, kernel_size=7, stride=1, padding=3, bias=False))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def _get_simple_model(mode, n_hidden_convs=1, hidden_channels=10, kernel_size=7):
    """Creates a simple CNN model that preserves spatial resolution. Depending on the mode, outputs 1 or 2 channels."""
    # Control on inputs
    assert mode in [Mode.MODE_T2X, Mode.MODE_T2XA, Mode.MODE_X2T,
                    Mode.MODE_X2TA], f"Mode {mode} invalid for {_get_simple_model.__name__}"
    assert kernel_size % 2 == 1, "Kernel size should be odd"

    # Model variables
    padding = int(kernel_size / 2 - .5)
    out_channels = 2 if mode in [Mode.MODE_T2XA, Mode.MODE_X2TA] else 1
    relu = nn.ReLU()

    # Model's input
    model = nn.Sequential(
        nn.Conv2d(1, hidden_channels, kernel_size, padding=padding),
        relu
    )

    # Hidden model convolutions
    for _ in range(n_hidden_convs):
        model.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding))
        model.append(relu)

    # Model output
    model.append(nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=padding))
    model.append(nn.Sigmoid())

    return model


def get_models(mode, n_hidden_convs=1, hidden_channels=10, kernel_size=7, device=None):
    """Creates a list of simple models based on the mode."""
    if mode == Mode.MODE_BOTH:
        return [
            _get_simple_model(Mode.MODE_T2X, n_hidden_convs, hidden_channels, kernel_size).to(device),
            _get_bottleneck_model(Mode.MODE_X2T, hidden_channels, n_hidden_convs).to(device)
        ]
    elif mode == Mode.MODE_BOTH_A:
        return [
            _get_simple_model(Mode.MODE_T2XA, n_hidden_convs, hidden_channels, kernel_size).to(device),
            _get_bottleneck_model(Mode.MODE_X2TA, hidden_channels, n_hidden_convs).to(device)
        ]
    elif mode in [Mode.MODE_X2T, Mode.MODE_X2TA]:
        return [_get_bottleneck_model(mode, hidden_channels, n_hidden_convs).to(device)]

    return [_get_simple_model(mode, n_hidden_convs, hidden_channels, kernel_size).to(device)]
