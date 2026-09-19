import importlib
import math
import os
import sys
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


T_MAX = 4096
_WKV_EXTENSION = None


def _load_wkv_extension():
    global _WKV_EXTENSION
    if _WKV_EXTENSION is not None:
        return _WKV_EXTENSION
    try:
        _WKV_EXTENSION = importlib.import_module("wkv")
        return _WKV_EXTENSION
    except ImportError:
        pass
    cache_root = os.path.expanduser("~/.cache/torch_extensions")
    if os.path.isdir(cache_root):
        for root, _, files in os.walk(cache_root):
            if os.path.basename(root) == "wkv" and "wkv.so" in files:
                sys.path.insert(0, root)
                try:
                    _WKV_EXTENSION = importlib.import_module("wkv")
                    return _WKV_EXTENSION
                except ImportError:
                    sys.path.pop(0)
    from torch.utils.cpp_extension import load

    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")
    _WKV_EXTENSION = load(
        name="wkv",
        sources=[
            os.path.join(source_dir, "wkv_op.cpp"),
            os.path.join(source_dir, "wkv_cuda.cu"),
        ],
        verbose=False,
        extra_cuda_cflags=[
            "-res-usage",
            "--maxrregcount",
            "60",
            "--use_fast_math",
            "-O3",
            "-Xptxas",
            "-O3",
            f"-DTmax={T_MAX}",
        ],
    )
    return _WKV_EXTENSION


def _torch_wkv(w, u, k, v):
    batch, tokens, channels = k.shape
    if tokens > 512:
        raise RuntimeError(
            "The PyTorch WKV fallback supports at most 512 tokens; use CUDA with a compiled WKV extension for full-resolution inference"
        )
    positions = torch.arange(tokens, device=k.device, dtype=k.dtype)
    distance = (positions[:, None] - positions[None, :]).abs()
    scores = k[:, None, :, :] - distance[None, :, :, None] * w[None, None, None, :]
    diagonal = torch.eye(tokens, device=k.device, dtype=k.dtype)
    scores = scores + diagonal[None, :, :, None] * u[None, None, None, :]
    weights = torch.softmax(scores.float(), dim=2).to(v.dtype)
    return (weights * v[:, None, :, :]).sum(dim=2)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, batch, tokens, channels, w, u, k, v):
        extension = _load_wkv_extension()
        ctx.batch = batch
        ctx.tokens = tokens
        ctx.channels = channels
        ctx.save_for_backward(w, u, k, v)
        half_mode = w.dtype == torch.float16
        bfloat_mode = w.dtype == torch.bfloat16
        w32 = w.float().contiguous()
        u32 = u.float().contiguous()
        k32 = k.float().contiguous()
        v32 = v.float().contiguous()
        output = torch.empty(
            (batch, tokens, channels),
            device=k.device,
            dtype=torch.float32,
            memory_format=torch.contiguous_format,
        )
        extension.forward(batch, tokens, channels, w32, u32, k32, v32, output)
        if half_mode:
            output = output.half()
        elif bfloat_mode:
            output = output.bfloat16()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        extension = _load_wkv_extension()
        batch = ctx.batch
        tokens = ctx.tokens
        channels = ctx.channels
        w, u, k, v = ctx.saved_tensors
        device = k.device
        grad_w = torch.zeros((batch, channels), device=device).contiguous()
        grad_u = torch.zeros((batch, channels), device=device).contiguous()
        grad_k = torch.zeros((batch, tokens, channels), device=device).contiguous()
        grad_v = torch.zeros((batch, tokens, channels), device=device).contiguous()
        extension.backward(
            batch,
            tokens,
            channels,
            w.float().contiguous(),
            u.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
            grad_output.float().contiguous(),
            grad_w,
            grad_u,
            grad_k,
            grad_v,
        )
        grad_w = grad_w.sum(dim=0).to(w.dtype)
        grad_u = grad_u.sum(dim=0).to(u.dtype)
        return None, None, None, grad_w, grad_u, grad_k.to(k.dtype), grad_v.to(v.dtype)


def RUN_CUDA(batch, tokens, channels, w, u, k, v):
    if k.is_cuda:
        return WKV.apply(batch, tokens, channels, w, u, k, v)
    return _torch_wkv(w, u, k, v)


class GELU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        del inplace

    def forward(self, x):
        return 0.5 * x * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def get_norm(norm_layer="bn_2d"):
    eps = 1e-6
    layers = {
        "none": nn.Identity,
        "bn_2d": partial(nn.BatchNorm2d, eps=eps),
        "ln_2d": partial(LayerNorm2d, eps=eps),
    }
    return layers[norm_layer]


def get_act(act_layer="relu"):
    layers = {
        "none": nn.Identity,
        "relu": nn.ReLU,
        "gelu": GELU,
        "silu": nn.SiLU,
    }
    return layers[act_layer]


class ConvNormAct(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        skip=False,
        norm_layer="bn_2d",
        act_layer="relu",
        inplace=True,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.norm = get_norm(norm_layer)(dim_out)
        self.act = get_act(act_layer)(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.act(self.norm(self.conv(x)))
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class SE(nn.Module):
    def __init__(self, in_chs, rd_ratio=0.25, act_layer=nn.ReLU):
        super().__init__()
        rd_channels = round(in_chs * rd_ratio)
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.act1(self.conv_reduce(x_se))
        return x * self.gate(self.conv_expand(x_se))


def q_shift(x, shift_pixel=1, gamma=0.25, patch_resolution=None):
    batch, _, channels = x.shape
    height, width = patch_resolution
    x = x.transpose(1, 2).reshape(batch, channels, height, width)
    output = torch.zeros_like(x)
    group = int(channels * gamma)
    output[:, 0:group, :, shift_pixel:width] = x[:, 0:group, :, 0 : width - shift_pixel]
    output[:, group : 2 * group, :, 0 : width - shift_pixel] = x[
        :, group : 2 * group, :, shift_pixel:width
    ]
    output[:, 2 * group : 3 * group, shift_pixel:height, :] = x[
        :, 2 * group : 3 * group, 0 : height - shift_pixel, :
    ]
    output[:, 3 * group : 4 * group, 0 : height - shift_pixel, :] = x[
        :, 3 * group : 4 * group, shift_pixel:height, :
    ]
    output[:, 4 * group :, :, :] = x[:, 4 * group :, :, :]
    return output.flatten(2).transpose(1, 2)


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, channel_gamma=0.25, shift_pixel=1):
        super().__init__()
        self.n_embd = n_embd
        self.spatial_decay = nn.Parameter(torch.zeros(n_embd))
        self.spatial_first = nn.Parameter(torch.zeros(n_embd))
        self.spatial_mix_k = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        self.spatial_mix_v = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        self.shift_pixel = shift_pixel
        self.channel_gamma = channel_gamma
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.key_norm = nn.LayerNorm(n_embd)
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x, patch_resolution=None):
        batch, tokens, channels = x.size()
        if self.shift_pixel > 0:
            shifted = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            key_input = x * self.spatial_mix_k + shifted * (1.0 - self.spatial_mix_k)
            value_input = x * self.spatial_mix_v + shifted * (1.0 - self.spatial_mix_v)
            receptance_input = x * self.spatial_mix_r + shifted * (1.0 - self.spatial_mix_r)
        else:
            key_input = value_input = receptance_input = x
        key = self.key(key_input)
        value = self.value(value_input)
        receptance = torch.sigmoid(self.receptance(receptance_input))
        output = RUN_CUDA(
            batch,
            tokens,
            channels,
            self.spatial_decay / tokens,
            self.spatial_first / tokens,
            key,
            value,
        )
        return self.output(receptance * self.key_norm(output))


class iR_RWKV(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_in=True,
        has_skip=True,
        exp_ratio=1.0,
        norm_layer="bn_2d",
        act_layer="relu",
        dw_ks=3,
        stride=1,
        dilation=1,
        se_ratio=0.0,
        attn_s=True,
        drop_path=0.0,
        drop=0.0,
        img_size=224,
        channel_gamma=0.25,
        shift_pixel=1,
    ):
        super().__init__()
        del img_size
        self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
        dim_mid = int(dim_in * exp_ratio)
        self.ln1 = nn.LayerNorm(dim_mid)
        self.conv = ConvNormAct(dim_in, dim_mid, kernel_size=1)
        self.has_skip = dim_in == dim_out and stride == 1 and has_skip
        if attn_s:
            self.att = VRWKV_SpatialMix(dim_mid, channel_gamma, shift_pixel)
        self.se = (
            SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer))
            if se_ratio > 0.0
            else nn.Identity()
        )
        self.proj_drop = nn.Dropout(drop)
        self.proj = ConvNormAct(
            dim_mid,
            dim_out,
            kernel_size=1,
            norm_layer="none",
            act_layer="none",
        )
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.attn_s = attn_s
        self.conv_local = ConvNormAct(
            dim_mid,
            dim_mid,
            kernel_size=dw_ks,
            stride=stride,
            dilation=dilation,
            groups=dim_mid,
            norm_layer="bn_2d",
            act_layer="silu",
        )

    def forward(self, x):
        shortcut = x
        x = self.conv(self.norm(x))
        if self.attn_s:
            batch, channels, height, width = x.size()
            tokens = x.view(batch, channels, -1).permute(0, 2, 1)
            tokens = tokens + self.drop_path(
                self.ln1(self.att(tokens, (height, width)))
            )
            x = tokens.permute(0, 2, 1).contiguous().view(
                batch, channels, height, width
            )
        local = self.se(self.conv_local(x))
        x = x + local if self.has_skip else local
        x = self.proj(self.proj_drop(x))
        return shortcut + self.drop_path(x) if self.has_skip else x


class RWKV_UNet_encoder(nn.Module):
    def __init__(
        self,
        dim_in=3,
        num_classes=1000,
        img_size=224,
        depths=(2, 4, 4, 2),
        stem_dim=16,
        embed_dims=(64, 128, 256, 512),
        exp_ratios=(2.0, 2.0, 4.0, 4.0),
        norm_layers=("bn_2d", "bn_2d", "bn_2d", "bn_2d"),
        act_layers=("relu", "relu", "relu", "relu"),
        dw_kss=(3, 3, 1, 1),
        se_ratios=(0.0, 0.0, 0.0, 0.0),
        attn_ss=(False, False, True, True),
        drop=0.0,
        drop_path=0.0,
        channel_gamma=0.25,
        shift_pixel=1,
    ):
        super().__init__()
        self.num_classes = num_classes
        rates = [value.item() for value in torch.linspace(0, drop_path, sum(depths))]
        self.stage0 = nn.ModuleList(
            [
                iR_RWKV(
                    dim_in,
                    stem_dim,
                    norm_in=False,
                    has_skip=False,
                    exp_ratio=1,
                    norm_layer=norm_layers[0],
                    act_layer=act_layers[0],
                    dw_ks=dw_kss[0],
                    stride=1,
                    se_ratio=1,
                    attn_s=False,
                    img_size=img_size,
                    shift_pixel=shift_pixel,
                )
            ]
        )
        img_size //= 2
        previous_dim = stem_dim
        for stage_index, depth in enumerate(depths):
            layers = []
            stage_rates = rates[sum(depths[:stage_index]) : sum(depths[: stage_index + 1])]
            for block_index in range(depth):
                if block_index == 0:
                    stride = 2
                    has_skip = False
                    use_attention = False
                    ratio = exp_ratios[stage_index] * 2
                    img_size //= 2
                else:
                    stride = 1
                    has_skip = True
                    use_attention = attn_ss[stage_index]
                    ratio = exp_ratios[stage_index]
                layers.append(
                    iR_RWKV(
                        previous_dim,
                        embed_dims[stage_index],
                        norm_in=True,
                        has_skip=has_skip,
                        exp_ratio=ratio,
                        norm_layer=norm_layers[stage_index],
                        act_layer=act_layers[stage_index],
                        dw_ks=dw_kss[stage_index],
                        stride=stride,
                        se_ratio=se_ratios[stage_index],
                        attn_s=use_attention,
                        drop_path=stage_rates[block_index],
                        drop=drop,
                        img_size=img_size,
                        channel_gamma=channel_gamma,
                        shift_pixel=shift_pixel,
                    )
                )
                previous_dim = embed_dims[stage_index]
            setattr(self, f"stage{stage_index + 1}", nn.ModuleList(layers))
        self.pre_dim = embed_dims[-1]
        self.norm = get_norm(norm_layers[-1])(embed_dims[-1])
        self.head = nn.Linear(self.pre_dim, num_classes)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(
            module,
            (
                nn.LayerNorm,
                nn.GroupNorm,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
            ),
        ):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)


def RWKV_UNet_encoder_B(pretrained=False, **kwargs):
    del pretrained
    return RWKV_UNet_encoder(
        depths=(3, 3, 6, 3),
        stem_dim=24,
        embed_dims=(48, 72, 144, 240),
        exp_ratios=(2.0, 2.5, 4.0, 4.0),
        norm_layers=("bn_2d", "bn_2d", "ln_2d", "ln_2d"),
        act_layers=("silu", "silu", "gelu", "gelu"),
        dw_kss=(5, 5, 5, 5),
        attn_ss=(False, False, True, True),
        drop=0.0,
        drop_path=0.05,
        **kwargs,
    )


__all__ = ["RUN_CUDA", "RWKV_UNet_encoder_B", "VRWKV_SpatialMix"]
