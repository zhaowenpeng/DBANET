import torch
import torch.nn as nn


def _powers(x, numerator_order, denominator_order):
    values = [x]
    for _ in range(max(numerator_order, denominator_order) - 2):
        values.append(values[-1] * x)
    values.insert(0, torch.ones_like(x))
    return torch.stack(values, dim=1)


def rational_group(x, numerator, denominator, groups):
    batch, length, channels = x.shape
    channels_per_group = channels // groups
    z = x.view(batch, length, groups, channels_per_group)
    z = z.permute(2, 0, 1, 3).contiguous()
    z = z.view(groups, batch * length * channels_per_group)
    powers = _powers(z, numerator.size(1), denominator.size(1))
    top = torch.bmm(numerator.unsqueeze(1), powers).squeeze(1)
    bottom_weights = torch.cat(
        (
            torch.ones(groups, 1, device=x.device, dtype=x.dtype),
            denominator,
            torch.zeros(
                groups,
                max(0, numerator.size(1) - denominator.size(1) - 1),
                device=x.device,
                dtype=x.dtype,
            ),
        ),
        dim=1,
    )
    bottom = torch.bmm(bottom_weights.abs().unsqueeze(1), powers).squeeze(1)
    result = top.div(bottom)
    result = result.view(groups, batch, length, channels_per_group)
    result = result.permute(1, 2, 0, 3).contiguous()
    return result.view(batch, length, channels)


class KATGroup(nn.Module):
    def __init__(self, num_groups=8, mode="swish"):
        super().__init__()
        if mode != "swish":
            raise ValueError("The standalone DBANet release supports the swish KAT initialization")
        self.order = (5, 4)
        self.num_groups = int(num_groups)
        numerator = [
            3.054879741161051e-07,
            0.5000007853744493,
            0.24999783422824703,
            0.05326628273219478,
            0.005803034571292244,
            0.0002751961022402342,
        ]
        denominator = [
            -4.111554955950634e-06,
            0.10652899335007572,
            -1.2690007399796238e-06,
            0.0005502331264140556,
        ]
        self.weight_numerator = nn.Parameter(torch.tensor(numerator).view(1, -1))
        self.weight_denominator = nn.Parameter(
            torch.tensor(denominator).repeat(self.num_groups, 1)
        )

    def forward(self, x):
        numerator = self.weight_numerator.repeat(self.num_groups, 1)
        return rational_group(x, numerator, self.weight_denominator, self.num_groups)


class GroupKANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_mode="swish",
        drop=0.0,
        use_conv=False,
        device=None,
        num_groups=8,
    ):
        super().__init__()
        del device
        self.act = KATGroup(num_groups=num_groups, mode=act_mode)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        if use_conv:
            self.linear = nn.Conv2d(in_features, out_features, kernel_size=1, bias=bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        original_dim = x.ndim
        if original_dim == 2:
            x = x.unsqueeze(1)
        x = self.act(x)
        x = self.drop(x)
        if original_dim == 2:
            x = x.squeeze(1)
        return self.linear(x)


__all__ = ["GroupKANLinear"]
