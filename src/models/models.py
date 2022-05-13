import torch.nn as nn
from utils.utils import Mode


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
            _get_simple_model(Mode.MODE_X2T, n_hidden_convs, hidden_channels, kernel_size).to(device)
        ]
    elif mode == Mode.MODE_BOTH_A:
        return [
            _get_simple_model(Mode.MODE_T2XA, n_hidden_convs, hidden_channels, kernel_size).to(device),
            _get_simple_model(Mode.MODE_X2TA, n_hidden_convs, hidden_channels, kernel_size).to(device)
        ]

    return [_get_simple_model(mode, n_hidden_convs, hidden_channels, kernel_size).to(device)]
