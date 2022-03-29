import torch
import torch.nn as nn
from torch.nn import functional as F


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


class ActNorm(nn.Module):
    """Activation Normalization class from GLOW pytorch implementation
    (https://github.com/rosinality/glow-pytorch/blob/master/model.py)"""

    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = torch.log(torch.abs(self.scale))

        if self.logdet:
            logdet = height * width * torch.sum(log_abs)
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine
        self.act_norm = ActNorm(in_channel)  # Edited by Brian Pulfer

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        input, log_in = self.act_norm(input)  # Edited by Brian Pulfer
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)  # + log_in  # Edited by Brian Pulfer

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class NormalizingFlowModel(nn.Module):
    def __init__(self, backbone, in_channels, n_layers=16, freeze_backbone=True):
        super(NormalizingFlowModel, self).__init__()

        self.backbone = backbone
        self.affine_layers = nn.ModuleList([AffineCoupling(in_channels) for _ in range(n_layers)])
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _fastflow(self, features):
        out, sum_log_det_js = features, 0
        for al in self.affine_layers:
            # Running affine coupling layer
            out, log_det_j = al(out)
            sum_log_det_js += log_det_j

            # Permutation of channels
            first_half, second_half = out.chunk(2, 1)
            out = torch.cat((second_half, first_half), dim=1)
        return out, sum_log_det_js

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
