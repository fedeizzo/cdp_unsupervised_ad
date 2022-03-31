import torch.nn as nn
from torchvision.models.resnet import resnet50


def adjust_resnet_input(resnet_fn, in_channels, pretrained=False):
    # Creating model
    resnet = resnet_fn(pretrained=pretrained)

    # Modifying first conv layer's expected input channels
    resnet.conv1 = nn.Conv2d(
        in_channels,
        resnet.conv1.out_channels,
        resnet.conv1.kernel_size,
        stride=resnet.conv1.stride,
        padding=resnet.conv1.padding,
        bias=resnet.conv1.bias
    )

    return resnet


def get_backbone_resnet(pretrained, final_n_channels):
    resnet = adjust_resnet_input(resnet50, in_channels=1, pretrained=pretrained)
    modules = list(resnet.children())[:-3]
    modules.append(nn.Conv2d(1024, final_n_channels, (1, 1)))  # Note: 1024 channels out of resnet50 (layer3)
    resnet = nn.Sequential(*modules)
    return resnet
