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


def get_backbone_resnet(resnet_fn, resnet_in_channels, resnet_out_channels, backbone_out_channels, pretrained,
                        resnet_layer=3):
    supported_layers = [1, 2, 3, 4]
    assert resnet_layer in supported_layers, \
        f"resnet_layer should be in {supported_layers}, found {resnet_layer} instead"

    resnet = adjust_resnet_input(resnet_fn, in_channels=resnet_in_channels, pretrained=pretrained)
    modules = list(resnet.children())[:-(2 + (4 - resnet_layer))]
    modules.append(nn.Conv2d(resnet_out_channels, backbone_out_channels, (1, 1)))
    resnet = nn.Sequential(*modules)
    return resnet
