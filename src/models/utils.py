import torch.nn as nn


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


def get_backbone_resnet(resnet_fn, resnet_out_channels, backbone_out_channels, pretrained):
    resnet = adjust_resnet_input(resnet_fn, in_channels=1, pretrained=pretrained)
    modules = list(resnet.children())[:-3]
    modules.append(nn.Conv2d(resnet_out_channels, backbone_out_channels, (1, 1)))
    resnet = nn.Sequential(*modules)
    return resnet
