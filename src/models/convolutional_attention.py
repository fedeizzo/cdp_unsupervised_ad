import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."

    def __init__(self, n_channels=8):
        super(SelfAttention, self).__init__()
        self.query, self.key, self.value = [
            self._conv(n_channels, c)
            for c in (n_channels // 8, n_channels // 8, n_channels)
        ]
        self.gamma = nn.parameter.Parameter(torch.Tensor([0.0]))

    def _conv(self, n_in, n_out):
        return nn.Conv1d(
            n_in,
            n_out,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class ConfidenceModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 8,
        kernel_size: int = 7,
        num_layers: int = 3,
        dropout: float = 0.2,
        padding=3,
    ):
        super(ConfidenceModel, self).__init__()
        self.expand = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.pool_size = 32
        self.adaptive_crop = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.reduce = nn.Conv2d(
            out_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.s_attn = SelfAttention()
        layers = []

        # channel * width * height
        in_linear = out_channels * self.pool_size * self.pool_size
        out_linear = (in_linear - 32) // num_layers
        for i in range(num_layers):
            layers += [
                nn.Linear(
                    in_linear - (out_linear * i),
                    in_linear - (out_linear * (i + 1)),
                ),
                nn.Dropout(dropout),
                nn.ReLU(),
            ]
        self.regressor = nn.Sequential(
            *layers,
            nn.Linear(32, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        expanded = self.expand(x)
        cropped = self.adaptive_crop(expanded)
        attn = self.s_attn(cropped)
        attn = attn.reshape(attn.shape[0], -1)
        out = self.regressor(attn)
        return out


if __name__ == "__main__":
    x = torch.randn([8, 1, 684, 684])
    c = ConfidenceModel(kernel_size=7)
    print(c(x).shape)
