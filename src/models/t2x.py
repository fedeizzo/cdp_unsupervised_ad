import torch.nn as nn


def get_t2x_model():
    return nn.Sequential(
        nn.Conv2d(1, 10, (7, 7), padding=(3, 3)),
        nn.Conv2d(10, 10, (7, 7), padding=(3, 3)),
        nn.ReLU(),
        nn.Conv2d(10, 1, (7, 7), padding=(3, 3)),
        nn.Sigmoid()
    )
