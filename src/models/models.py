import torch
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


class AffineCouplingLayer(nn.Module):
    def __init__(self, in_channels, first_half=True):
        super(AffineCouplingLayer, self).__init__()

        # Attributes
        assert (in_channels % 2 == 0), f"Input channels for an AffineCouplingLayer must be even, but got {in_channels}"
        self.first_half = first_half
        self.half = in_channels // 2

        # Functions
        # TODO: Use ActNorm instead of BatchNorm --> https://paperswithcode.com/method/activation-normalization
        self.act_norm = nn.BatchNorm2d(in_channels)
        self.s, self.t = self._get_conv_block(self.half), self._get_conv_block(self.half)

    def forward(self, x, prev_layer_det=None):
        # Checking input dimensions, converting to batch (N, C, H, W) and splitting input tensors
        assert x.dim() >= 3
        if x.dim == 3:
            x = x.unsqueeze(0)

        # Normalizing input
        x = self.act_norm(x)

        # Splitting in halves
        first = x[:, :self.half, :, :]
        second = x[:, self.half:, :, :]

        # Running forward
        if self.first_half:
            scale, translation = self.s(second), self.t(second)
            first = first * torch.exp(scale) + translation
        else:
            scale, translation = self.s(first), self.t(first)
            second = second * torch.exp(scale) + translation

        log_det_j = torch.sum(scale, dim=[1, 2, 3])

        if prev_layer_det is not None:
            log_det_j += prev_layer_det

        return torch.cat((first, second), dim=1), log_det_j

    @staticmethod
    def _get_conv_block(in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0))
        )


class NormalizingFlowModel(nn.Module):
    def __init__(self, backbone, in_channels, n_layers=4):
        super(NormalizingFlowModel, self).__init__()

        self.backbone = backbone
        self.affine_layers = [AffineCouplingLayer(in_channels, i % 2 == 0) for i in range(n_layers)]

    def _fastflow(self, features):
        out, log_det_j = features, None
        for al in self.affine_layers:
            out, log_det_j = al(out, log_det_j)
        return out, log_det_j

    def forward(self, x):
        # Checking dimensionality
        assert x.dim() >= 3
        if x.dim == 3:
            x = x.unsqueeze(0)

        # Getting backbone features
        features = self.backbone(x)

        # Getting affine-coupling layers outputs
        out, log_det_j = self._fastflow(features)

        return features, out, log_det_j
