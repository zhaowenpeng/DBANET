from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .ikan import GroupKANLinear as _GroupKANLinear
    from .rwkv_unet import RUN_CUDA, RWKV_UNet_encoder_B, VRWKV_SpatialMix
except ImportError:
    from ikan import GroupKANLinear as _GroupKANLinear
    from rwkv_unet import RUN_CUDA, RWKV_UNet_encoder_B, VRWKV_SpatialMix

_HAS_GKAN = True

Tensor = torch.Tensor
KernelSize = Union[int, Tuple[int, int]]
_BASE_GRID_CACHE: Dict[Tuple[int, int, str, torch.dtype, bool], Tensor] = {}


def _monitoring_enabled(module: nn.Module) -> bool:
    return bool(getattr(module, "_monitoring_enabled", True))


def _norm2d(channels: int) -> nn.Module:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _padding(
    kernel_size: KernelSize,
    dilation: int = 1,
) -> Union[int, Tuple[int, int]]:
    if isinstance(kernel_size, tuple):
        return tuple(dilation * (size // 2) for size in kernel_size)
    return dilation * (kernel_size // 2)


class ConvNormAct(nn.Sequential):
                                   
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: KernelSize = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activate: bool = True,
    ) -> None:
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=_padding(kernel_size, dilation),
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            _norm2d(out_channels),
        ]
        if activate:
            layers.append(nn.GELU())
        super().__init__(*layers)


class DepthwiseSeparableConv(nn.Module):
                                 
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: KernelSize = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


if _HAS_GKAN:
    class GRKANBlock(nn.Module):
                                                  

        def __init__(
            self,
            channels: int,
            hidden_channels: Optional[int] = None,
            dropout: float = 0.1,
            num_groups: int = 8,
        ) -> None:
            super().__init__()
            hidden_channels = hidden_channels or channels
            self.norm = nn.LayerNorm(channels)
            self.fc1 = _GroupKANLinear(
                channels,
                hidden_channels,
                bias=True,
                act_mode="swish",
                drop=dropout,
                use_conv=False,
                num_groups=num_groups,
            )
            self.fc2 = _GroupKANLinear(
                hidden_channels,
                channels,
                bias=True,
                act_mode="swish",
                drop=dropout,
                use_conv=False,
                num_groups=num_groups,
            )
            self.fallback = nn.Sequential(
                nn.Linear(channels, hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, channels),
                nn.Dropout(dropout),
            )
            self.scale = nn.Parameter(torch.ones(1, 1, channels) * 1e-3)

        def forward(self, x: Tensor) -> Tensor:
            x = self.norm(x)
            batch, tokens, channels = x.shape
            if x.is_cuda:
                x = self.fc1(x.reshape(batch * tokens, channels)).reshape(
                    batch,
                    tokens,
                    -1,
                )
                x = self.fc2(x.reshape(batch * tokens, -1)).reshape(
                    batch,
                    tokens,
                    channels,
                )
            else:
                x = self.fallback(x)
            return self.scale * x
else:
    class GRKANBlock(nn.Module):
                                          

        def __init__(
            self,
            channels: int,
            hidden_channels: Optional[int] = None,
            dropout: float = 0.1,
            **_: object,
        ) -> None:
            super().__init__()
            hidden_channels = hidden_channels or channels
            self.norm = nn.LayerNorm(channels)
            self.net = nn.Sequential(
                nn.Linear(channels, hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, channels),
                nn.Dropout(dropout),
            )
            self.scale = nn.Parameter(torch.ones(1, 1, channels) * 1e-3)

        def forward(self, x: Tensor) -> Tensor:
            return self.scale * self.net(self.norm(x))


def _base_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    align_corners: bool = True,
) -> Tensor:
    key = (
        height,
        width,
        str(device),
        dtype,
        bool(align_corners),
    )
    cached = _BASE_GRID_CACHE.get(key)
    if cached is not None:
        return cached
    if align_corners:
        ys = torch.linspace(
            -1.0,
            1.0,
            height,
            device=device,
            dtype=dtype,
        )
        xs = torch.linspace(
            -1.0,
            1.0,
            width,
            device=device,
            dtype=dtype,
        )
    else:
        ys = (
            (torch.arange(height, device=device, dtype=dtype) + 0.5)
            * (2.0 / height)
            - 1.0
        )
        xs = (
            (torch.arange(width, device=device, dtype=dtype) + 0.5)
            * (2.0 / width)
            - 1.0
        )
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
    _BASE_GRID_CACHE[key] = grid
    return grid


class FoundationSemanticEncoder(nn.Module):
                                          

    def __init__(
        self,
        input_size: int = 256,
        num_prompts: int = 4,
        pretrained: bool = True,
        model_name: str = "vit_large_patch16_dinov3.sat493m",
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            img_size=self.input_size,
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

        self.embed_dim = int(self.backbone.embed_dim)
        self.num_prefix = int(self.backbone.num_prefix_tokens)
        self.num_prompts = int(num_prompts)
        self.prompts = nn.Parameter(
            torch.empty(1, self.num_prompts, self.embed_dim)
        )
        nn.init.trunc_normal_(self.prompts, std=0.02)

    def train(self, mode: bool = True) -> "FoundationSemanticEncoder":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        backbone = self.backbone
        with torch.no_grad():
            tokens = backbone.patch_embed(x)
            positioned = backbone._pos_embed(tokens)
            if isinstance(positioned, tuple):
                tokens, rope = positioned
            else:
                tokens, rope = positioned, None
            if getattr(backbone, "patch_drop", None) is not None:
                tokens = backbone.patch_drop(tokens)
            if getattr(backbone, "norm_pre", None) is not None:
                tokens = backbone.norm_pre(tokens)

        prefix = tokens[:, : self.num_prefix]
        patches = tokens[:, self.num_prefix :]
        prompts = self.prompts.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((prefix, prompts, patches), dim=1)

        attention_modules = [
            block.attn
            for block in backbone.blocks
            if hasattr(block, "attn")
            and hasattr(block.attn, "num_prefix_tokens")
        ]
        original_prefix_counts = [
            module.num_prefix_tokens for module in attention_modules
        ]
        for module in attention_modules:
            module.num_prefix_tokens = self.num_prefix + self.num_prompts

        try:
            for block in backbone.blocks:
                try:
                    tokens = block(tokens, rope=rope)
                except TypeError:
                    tokens = block(tokens)
        finally:
            for module, count in zip(
                attention_modules,
                original_prefix_counts,
            ):
                module.num_prefix_tokens = count

        tokens = backbone.norm(tokens)
        patches = tokens[:, self.num_prefix + self.num_prompts :]
        side = int(patches.shape[1] ** 0.5)
        if side * side != patches.shape[1]:
            raise RuntimeError(
                "DINO patch tokens do not form a square feature map: "
                f"{patches.shape[1]} tokens"
            )
        return patches.transpose(1, 2).reshape(
            x.shape[0],
            self.embed_dim,
            side,
            side,
        )


class MultiDepthDINOv3SemanticEncoder(nn.Module):
\
\
\
\
\
\
\
\
\
       

    def __init__(
        self,
        input_size: int = 512,
        num_prompts: int = 4,
        pretrained: bool = True,
        model_name: str = "vit_large_patch16_dinov3.sat493m",
        layer_indices: Sequence[int] = (4, 11, 17, 23),
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            img_size=self.input_size,
        )

                                         
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

        self.embed_dim = int(self.backbone.embed_dim)
        self.num_prefix = int(self.backbone.num_prefix_tokens)
        self.num_prompts = int(num_prompts)
        self.layer_indices = tuple(int(index) for index in layer_indices)
        if not self.layer_indices:
            raise ValueError("layer_indices must contain at least one DINO block")
        if tuple(sorted(set(self.layer_indices))) != self.layer_indices:
            raise ValueError("layer_indices must be unique and sorted")
        if self.layer_indices[-1] >= len(self.backbone.blocks):
            raise ValueError(
                f"DINO has {len(self.backbone.blocks)} blocks, but layer "
                f"{self.layer_indices[-1]} was requested"
            )

        self.prompts = nn.Parameter(
            torch.empty(1, self.num_prompts, self.embed_dim)
        )
        nn.init.trunc_normal_(self.prompts, std=0.02)

                                               
                                       
        self.intermediate_norms = nn.ModuleList(
            nn.LayerNorm(self.embed_dim)
            for _ in self.layer_indices[:-1]
        )

    def train(
        self,
        mode: bool = True,
    ) -> "MultiDepthDINOv3SemanticEncoder":
        super().train(mode)
                                                 
                                       
        self.backbone.eval()
        return self

    @staticmethod
    def _to_feature_map(patches: Tensor) -> Tensor:
                                                    
        side = int(patches.shape[1] ** 0.5)
        if side * side != patches.shape[1]:
            raise RuntimeError(
                "DINO patch tokens do not form a square feature map: "
                f"{patches.shape[1]} tokens"
            )
        return patches.transpose(1, 2).reshape(
            patches.shape[0],
            patches.shape[2],
            side,
            side,
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

        backbone = self.backbone
                                                   
                                    
        with torch.no_grad():
            tokens = backbone.patch_embed(x)
            positioned = backbone._pos_embed(tokens)
            if isinstance(positioned, tuple):
                tokens, rope = positioned
            else:
                tokens, rope = positioned, None
            if getattr(backbone, "patch_drop", None) is not None:
                tokens = backbone.patch_drop(tokens)
            if getattr(backbone, "norm_pre", None) is not None:
                tokens = backbone.norm_pre(tokens)

        prefix = tokens[:, : self.num_prefix]
        patches = tokens[:, self.num_prefix :]
        prompts = self.prompts.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((prefix, prompts, patches), dim=1)

                                                            
                                                   
        attention_modules = [
            block.attn
            for block in backbone.blocks
            if hasattr(block, "attn")
            and hasattr(block.attn, "num_prefix_tokens")
        ]
        original_prefix_counts = [
            module.num_prefix_tokens for module in attention_modules
        ]
        for module in attention_modules:
            module.num_prefix_tokens = self.num_prefix + self.num_prompts

        selected: List[Tensor] = []
        selected_set = set(self.layer_indices)
        try:
            for index, block in enumerate(backbone.blocks):
                try:
                    tokens = block(tokens, rope=rope)
                except TypeError:
                    tokens = block(tokens)
                if index in selected_set:
                    selected.append(tokens)
        finally:
            for module, count in zip(
                attention_modules,
                original_prefix_counts,
            ):
                module.num_prefix_tokens = count

        patch_start = self.num_prefix + self.num_prompts
        features: List[Tensor] = []
        for norm, layer_tokens in zip(
            self.intermediate_norms,
            selected[:-1],
        ):
            normalized = norm(layer_tokens[:, patch_start:])
            features.append(self._to_feature_map(normalized))
        final_tokens = backbone.norm(selected[-1])
        features.append(
            self._to_feature_map(final_tokens[:, patch_start:])
        )
        return features


class AnchorPreservingMultiDepthFusion(nn.Module):
\
\
\
\
\
       

    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(
                "multi-depth fusion requires at least two features"
            )
        self.final_projection = ConvNormAct(
            embed_dim,
            out_channels,
            kernel_size=1,
        )
        self.intermediate_projections = nn.ModuleList(
            ConvNormAct(embed_dim, out_channels, kernel_size=1)
            for _ in range(num_layers - 1)
        )
        self.layer_logits = nn.Parameter(torch.zeros(num_layers - 1))
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self._mon: Dict[str, object] = {}

    def forward(
        self,
        features: Sequence[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if len(features) != len(self.intermediate_projections) + 1:
            raise ValueError(
                f"expected {len(self.intermediate_projections) + 1} DINO "
                f"features, received {len(features)}"
            )

        semantic_anchor = self.final_projection(features[-1])
        weights = self.layer_logits.softmax(dim=0)
        intermediate = torch.zeros_like(semantic_anchor)
        for weight, projection, feature in zip(
            weights,
            self.intermediate_projections,
            features[:-1],
        ):
            projected = projection(feature)
            if projected.shape[-2:] != semantic_anchor.shape[-2:]:
                projected = F.interpolate(
                    projected,
                    size=semantic_anchor.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            intermediate = intermediate + weight * projected

        gate = torch.tanh(self.residual_gate)
        foundation = semantic_anchor + gate * intermediate
        if _monitoring_enabled(self):
            with torch.no_grad():
                self._mon = {
                    "weights": [
                        float(value) for value in weights.detach()
                    ],
                    "gate": float(gate.detach()),
                    "residual_ratio": float(
                        (gate * intermediate).abs().mean()
                        / (semantic_anchor.abs().mean() + 1e-8)
                    ),
                }
        return foundation, semantic_anchor


class SnakeShapeBranch(nn.Module):
                                 

    def __init__(
        self,
        channels: int,
        strip_kernel: int = 7,
    ) -> None:
        super().__init__()
        hidden = max(channels // 2, 16)
        self.offset_head = nn.Sequential(
            ConvNormAct(channels, hidden, kernel_size=3),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)

        self.horizontal = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, strip_kernel),
            padding=(0, strip_kernel // 2),
            groups=channels,
            bias=False,
        )
        self.vertical = nn.Conv2d(
            channels,
            channels,
            kernel_size=(strip_kernel, 1),
            padding=(strip_kernel // 2, 0),
            groups=channels,
            bias=False,
        )
        self.fuse = ConvNormAct(channels, channels, kernel_size=1)
        self.curve_head = nn.Conv2d(channels, 1, kernel_size=1)
        self._mon: Dict[str, float] = {"off": 0.0, "delta": 0.0}

    def forward(
        self,
        x: Tensor,
        return_curve: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        batch, _, height, width = x.shape
        offsets = self.offset_head(x)
        base = _base_grid(height, width, x.device, torch.float32)
        dx = offsets[:, 0].float() * (2.0 / max(width - 1, 1))
        dy = offsets[:, 1].float() * (2.0 / max(height - 1, 1))
        grid = base + torch.stack((dx, dy), dim=-1)
        warped = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        delta = self.fuse(
            self.horizontal(warped) + self.vertical(warped)
        )
        curve = self.curve_head(delta) if return_curve else None
        if _monitoring_enabled(self):
            with torch.no_grad():
                self._mon = {
                    "off": float(offsets.detach().abs().mean()),
                    "delta": float(delta.detach().abs().mean()),
                }
        return delta, curve


def bounded_directional_shift(
    x: Tensor,
    offsets: Tensor,
    gamma: float = 0.25,
    shift_pixel: int = 1,
) -> Tensor:
                                                                                 

    batch, channels, height, width = x.shape
    del batch
    group_channels = int(channels * gamma)
    sx = 2.0 / max(width - 1, 1)
    sy = 2.0 / max(height - 1, 1)
    base = _base_grid(height, width, x.device, torch.float32)
    deformed = base + torch.stack(
        (
            offsets[:, 0].float() * sx,
            offsets[:, 1].float() * sy,
        ),
        dim=-1,
    )
    directions = (
        (-shift_pixel * sx, 0.0),
        (shift_pixel * sx, 0.0),
        (0.0, -shift_pixel * sy),
        (0.0, shift_pixel * sy),
    )
    grouped_inputs: List[Tensor] = []
    grouped_grids: List[Tensor] = []
    for index, (dx, dy) in enumerate(directions):
        start = index * group_channels
        end = (index + 1) * group_channels
        grouped_inputs.append(x[:, start:end])
        grouped_grids.append(deformed + torch.tensor(
            (dx, dy),
            device=x.device,
            dtype=torch.float32,
        ))
    sampled = F.grid_sample(
        torch.cat(grouped_inputs, dim=0),
        torch.cat(grouped_grids, dim=0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    shifted_groups = list(sampled.chunk(len(directions), dim=0))
    if 4 * group_channels < channels:
        shifted_groups.append(x[:, 4 * group_channels :])
    return torch.cat(shifted_groups, dim=1)


class DeformableTokenShiftWKV(nn.Module):
                                                                            

    def __init__(
        self,
        attention: nn.Module,
    ) -> None:
        super().__init__()
        self.attention = attention
        channels = int(attention.n_embd)
        hidden = max(channels // 4, 16)
        self.offset_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        self.column_gate = nn.Parameter(torch.zeros(1))
        self._mon: Dict[str, float] = {"off": 0.0, "col": 0.0}

    def forward(
        self,
        x: Tensor,
        patch_resolution: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if patch_resolution is None:
            raise ValueError("Deformable WKV requires patch_resolution")
        attention = self.attention
        batch, tokens, channels = x.shape
        height, width = patch_resolution

        x_2d = x.transpose(1, 2).reshape(
            batch,
            channels,
            height,
            width,
        )
        offsets = self.offset_head(x_2d)
        shifted = bounded_directional_shift(
            x_2d,
            offsets,
            attention.channel_gamma,
            attention.shift_pixel,
        )
        shifted = shifted.flatten(2).transpose(1, 2)

        key_input = (
            x * attention.spatial_mix_k
            + shifted * (1.0 - attention.spatial_mix_k)
        )
        value_input = (
            x * attention.spatial_mix_v
            + shifted * (1.0 - attention.spatial_mix_v)
        )
        receptance_input = (
            x * attention.spatial_mix_r
            + shifted * (1.0 - attention.spatial_mix_r)
        )
        key = attention.key(key_input)
        value = attention.value(value_input)
        receptance = torch.sigmoid(
            attention.receptance(receptance_input)
        )

        row = RUN_CUDA(
            batch,
            tokens,
            channels,
            attention.spatial_decay / tokens,
            attention.spatial_first / tokens,
            key,
            value,
        )
        key_column = (
            key.reshape(batch, height, width, channels)
            .transpose(1, 2)
            .reshape(batch, tokens, channels)
        )
        value_column = (
            value.reshape(batch, height, width, channels)
            .transpose(1, 2)
            .reshape(batch, tokens, channels)
        )
        column = RUN_CUDA(
            batch,
            tokens,
            channels,
            attention.spatial_decay / tokens,
            attention.spatial_first / tokens,
            key_column,
            value_column,
        )
        column = (
            column.reshape(batch, width, height, channels)
            .transpose(1, 2)
            .reshape(batch, tokens, channels)
        )
        mixed = row + torch.tanh(self.column_gate) * column
        mixed = attention.key_norm(mixed)
        mixed = receptance * mixed
        if _monitoring_enabled(self):
            with torch.no_grad():
                self._mon = {
                    "off": float(offsets.detach().abs().mean()),
                    "col": float(torch.tanh(self.column_gate).detach()),
                }
        return attention.output(mixed)


class HierarchicalShapeAwareRWKVEncoder(nn.Module):
\
\
\
\
\
\
       

    DIMS: Sequence[int] = (24, 48, 72, 144, 240)
    PRETRAINED_URL = (
        "https://huggingface.co/FengheTan9/U-Stone/resolve/main/net_B.pth"
    )

    def __init__(
        self,
        image_size: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = RWKV_UNet_encoder_B(img_size=image_size)
        if pretrained:
            self._load_pretrained()

        self.shape_e1 = SnakeShapeBranch(self.DIMS[1])
        self.shape_e2 = SnakeShapeBranch(self.DIMS[2])
        self._snake_curve: List[Tensor] = []
        for stage in (self.encoder.stage3, self.encoder.stage4):
            for block in stage:
                if isinstance(
                    getattr(block, "att", None),
                    VRWKV_SpatialMix,
                ):
                    block.att = DeformableTokenShiftWKV(
                        block.att
                    )

    def _load_pretrained(self) -> None:
        from torch.hub import load_state_dict_from_url

        try:
            state = load_state_dict_from_url(
                self.PRETRAINED_URL,
                progress=True,
                map_location="cpu",
            )
            missing, unexpected = self.encoder.load_state_dict(
                state,
                strict=False,
            )
            print(
                "[DBANet/HSRE] RWKV pretrained weights loaded: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        except Exception as error:
            raise RuntimeError(
                "DBANet requires the RWKV-UNet-B pretrained checkpoint. "
                f"Loading failed: {error}"
            ) from error

    @staticmethod
    def _run_stage(stage: nn.ModuleList, x: Tensor) -> Tensor:
        for block in stage:
            x = block(x)
        return x

    def forward(
        self,
        x: Tensor,
        return_shape_logits: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        features: List[Tensor] = []
        for stage in (
            self.encoder.stage0,
            self.encoder.stage1,
            self.encoder.stage2,
            self.encoder.stage3,
            self.encoder.stage4,
        ):
            x = self._run_stage(stage, x)
            features.append(x)
        delta_e1, curve_e1 = self.shape_e1(
            features[1],
            return_curve=return_shape_logits,
        )
        delta_e2, curve_e2 = self.shape_e2(
            features[2],
            return_curve=return_shape_logits,
        )
        features[1] = features[1] + delta_e1
        features[2] = features[2] + delta_e2
        self._snake_curve = (
            [curve_e1, curve_e2] if return_shape_logits else []
        )
        return tuple(features)

    def shift_wrappers(self):
                                              
        for stage in (self.encoder.stage3, self.encoder.stage4):
            for block in stage:
                if isinstance(
                    getattr(block, "att", None),
                    DeformableTokenShiftWKV,
                ):
                    yield block.att


                                        
ShapeAwareRWKVEncoder = HierarchicalShapeAwareRWKVEncoder


class ConflictGatedCrossGuidance(nn.Module):
                                     

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.peer_proj = ConvNormAct(
            channels,
            channels,
            kernel_size=1,
        )
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, query: Tensor, peer: Tensor) -> Tensor:
        peer = self.peer_proj(peer)
        gate = self.gate(torch.cat((query, peer), dim=1))
        return query + gate * peer


class SoftFrequencyDecoupling(nn.Module):
                               

    def __init__(
        self,
        channels: int,
        temperature: float = 0.08,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.split_logit = nn.Parameter(torch.tensor(0.0))
        self.region_proj = ConvNormAct(
            channels,
            channels,
            kernel_size=1,
        )
        self.boundary_proj = ConvNormAct(
            channels,
            channels,
            kernel_size=1,
        )
        self._radius_cache: Dict[
            Tuple[int, int, str],
            Tensor,
        ] = {}
        self._mon: Dict[str, float] = {}

    @torch.no_grad()
    def _radius(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        key = (height, width, str(device))
        if key not in self._radius_cache:
            fy = torch.fft.fftfreq(height, device=device)
            fx = torch.fft.rfftfreq(width, device=device)
            grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
            radius = torch.sqrt(grid_y.square() + grid_x.square())
            self._radius_cache[key] = radius / (
                radius.max() + 1e-6
            )
        return self._radius_cache[key]

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        height, width = x.shape[-2:]
        with torch.amp.autocast(
            device_type=x.device.type,
            enabled=False,
        ):
            spectrum = torch.fft.rfft2(
                x.float(),
                dim=(-2, -1),
                norm="ortho",
            )
            radius = self._radius(height, width, x.device)
            split = torch.sigmoid(self.split_logit)
            low_mask = torch.sigmoid(
                (split - radius) / self.temperature
            )
            high_mask = 1.0 - low_mask
            region = torch.fft.irfft2(
                spectrum * low_mask,
                s=(height, width),
                dim=(-2, -1),
                norm="ortho",
            )
            boundary = torch.fft.irfft2(
                spectrum * high_mask,
                s=(height, width),
                dim=(-2, -1),
                norm="ortho",
            )
            if _monitoring_enabled(self):
                with torch.no_grad():
                    amplitude = spectrum.abs()
                    low_energy = float((amplitude * low_mask).mean())
                    high_energy = float((amplitude * high_mask).mean())
                    self._mon = {
                        "split": float(split.detach()),
                        "low_e": low_energy,
                        "high_e": high_energy,
                        "hi_ratio": high_energy / (low_energy + high_energy + 1e-8),
                    }
        return (
            self.region_proj(region.to(x.dtype)),
            self.boundary_proj(boundary.to(x.dtype)),
        )


class FoundationGuidedDBRD(nn.Module):
                                        

    def __init__(self, channels: int = 192) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            ConvNormAct(3 * channels, channels, kernel_size=3),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )
        self.frequency = SoftFrequencyDecoupling(channels)
        self.boundary_from_region = ConflictGatedCrossGuidance(
            channels
        )
        self.region_from_boundary = ConflictGatedCrossGuidance(
            channels
        )
        self.reconstruction_refine = GRKANBlock(
            channels,
            hidden_channels=channels * 2,
        )
        self.region_polish = ConvNormAct(
            channels,
            channels,
            kernel_size=3,
        )
        self._mon: Dict[str, float] = {}

    def forward(
        self,
        foundation: Tensor,
        task_e3: Tensor,
        task_e4: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        fused = self.fuse(
            torch.cat((foundation, task_e3, task_e4), dim=1)
        )
        cosine = (
            F.normalize(foundation, dim=1)
            * F.normalize(task_e3, dim=1)
        ).sum(dim=1, keepdim=True)
        disagreement = ((1.0 - cosine) * 0.5).clamp(0.0, 1.0)

        region, boundary = self.frequency(fused)
        boundary = boundary * (1.0 + disagreement)
        boundary = self.boundary_from_region(boundary, region)
        region = self.region_from_boundary(region, boundary)
        batch, channels, height, width = region.shape
        region_delta = self.reconstruction_refine(
            region.flatten(2).transpose(1, 2)
        ).transpose(1, 2).reshape(batch, channels, height, width)
        region = region + region_delta
        region = self.region_polish(region)
        if _monitoring_enabled(self):
            with torch.no_grad():
                self._last_kan_effect = float(
                    region_delta.abs().mean()
                    / (region.abs().mean() + 1e-8)
                )
                self._mon = {
                    "dis": float(disagreement.detach().mean()),
                    "fused": float(fused.detach().abs().mean()),
                    "region": float(region.detach().abs().mean()),
                    "boundary": float(boundary.detach().abs().mean()),
                    "kan_effect": self._last_kan_effect,
                    **self.frequency._mon,
                }
        return region, boundary, disagreement


class DINOv3GuidedBoundaryRegionDecouplingRefinement(
    FoundationGuidedDBRD
):
\
\
\
\
\
\
\
\
\
\
       

    def forward(
        self,
        foundation: Tensor,
        semantic_anchor: Tensor,
        task_e3: Tensor,
        task_e4: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
                                        
        fused = self.fuse(
            torch.cat((foundation, task_e3, task_e4), dim=1)
        )

                                        
                                       
        cosine = (
            F.normalize(semantic_anchor, dim=1)
            * F.normalize(task_e3, dim=1)
        ).sum(dim=1, keepdim=True)
        disagreement = ((1.0 - cosine) * 0.5).clamp(0.0, 1.0)

        region, boundary = self.frequency(fused)
        boundary = boundary * (1.0 + disagreement)

                                         
                                 
        boundary = self.boundary_from_region(boundary, region)
        region = self.region_from_boundary(region, boundary)

                                                   
        batch, channels, height, width = region.shape
        region_delta = self.reconstruction_refine(
            region.flatten(2).transpose(1, 2)
        ).transpose(1, 2).reshape(batch, channels, height, width)
        region = self.region_polish(region + region_delta)

        if _monitoring_enabled(self):
            with torch.no_grad():
                self._last_kan_effect = float(
                    region_delta.abs().mean()
                    / (region.abs().mean() + 1e-8)
                )
                self._mon = {
                    "dis": float(disagreement.detach().mean()),
                    "fused": float(fused.detach().abs().mean()),
                    "region": float(region.detach().abs().mean()),
                    "boundary": float(boundary.detach().abs().mean()),
                    "kan_effect": self._last_kan_effect,
                    **self.frequency._mon,
                }
        return region, boundary, disagreement


class DynamicSampleUpsampler(nn.Module):
                           

    def __init__(self, channels: int, scale: int = 2) -> None:
        super().__init__()
        self.scale = int(scale)
        self.scope = 0.25
        hidden = max(channels // 4, 16)
        self.offset_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        self._mon: Dict[str, float] = {"off": 0.0}

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        offset = F.interpolate(
            self.offset_head(x),
            scale_factor=self.scale,
            mode="bilinear",
            align_corners=False,
        )
        grid = _base_grid(
            self.scale * height,
            self.scale * width,
            x.device,
            x.dtype,
            align_corners=False,
        ).expand(batch, -1, -1, -1)
        grid = grid + offset.permute(0, 2, 3, 1) * self.scope
        if _monitoring_enabled(self):
            with torch.no_grad():
                self._mon = {"off": float(offset.detach().abs().mean())}
        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )


class MultiScaleStripContext(nn.Module):
                                      

    def __init__(
        self,
        channels: int,
        strip_kernels: Sequence[int] = (7, 11, 21),
    ) -> None:
        super().__init__()
        self.local = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=False,
        )
        self.strips = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(1, kernel),
                        padding=(0, kernel // 2),
                        groups=channels,
                        bias=False,
                    ),
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(kernel, 1),
                        padding=(kernel // 2, 0),
                        groups=channels,
                        bias=False,
                    ),
                )
                for kernel in strip_kernels
            ]
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self._mon: Dict[str, float] = {"gamma": 0.0, "attn": 0.0}

    def forward(self, x: Tensor) -> Tensor:
        context = self.local(x)
        context = context + sum(branch(context) for branch in self.strips)
        attention = torch.sigmoid(self.project(context))
        if _monitoring_enabled(self):
            with torch.no_grad():
                self._mon = {
                    "gamma": float(self.residual_scale.detach()),
                    "attn": float(attention.detach().mean()),
                }
        return x + self.residual_scale * (x * attention)


class MultiScaleBoundaryAwareDecoderStage(nn.Module):
\
\
\
\
\
       

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        boundary_prior_channels: int,
        boundary_projection_channels: int = 32,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.use_skip = skip_channels > 0
        self.up = DynamicSampleUpsampler(in_channels)
        self.boundary_projection = ConvNormAct(
            boundary_prior_channels,
            boundary_projection_channels,
            kernel_size=1,
        )
        merge_channels = (
            in_channels
            + (skip_channels if self.use_skip else 0)
            + boundary_projection_channels
        )
        self.merge = nn.Sequential(
            ConvNormAct(merge_channels, out_channels),
            ConvNormAct(out_channels, out_channels),
        )
        self.context = MultiScaleStripContext(out_channels)
        self.segmentation_head = nn.Conv2d(
            out_channels,
            num_classes,
            kernel_size=1,
        )
        self.boundary_head = nn.Conv2d(
            out_channels,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        x: Tensor,
        skip: Optional[Tensor],
        boundary_prior: Tensor,
        return_aux: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        x = self.up(x)
        target_size = x.shape[-2:]
        parts = [x]
        if self.use_skip:
            if skip is None:
                raise ValueError("MBD stage requires a skip feature")
            if skip.shape[-2:] != target_size:
                skip = F.interpolate(
                    skip,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            parts.append(skip)
        boundary = F.interpolate(
            boundary_prior,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        parts.append(self.boundary_projection(boundary))
        x = self.context(self.merge(torch.cat(parts, dim=1)))
        if not return_aux:
            return x, None, None
        return x, self.segmentation_head(x), self.boundary_head(x)


                                           
MBDStage = MultiScaleBoundaryAwareDecoderStage


class DBANet(nn.Module):
\
\
\
\
\
\
\
\
\
\
\
       

    DINO_LAYERS: Sequence[int] = (4, 11, 17, 23)

    def __init__(
        self,
        num_classes: int = 1,
        img_size: int = 512,
        dino_img_size: Optional[int] = None,
        pretrained: bool = True,
        num_prompts: int = 4,
        neck_channels: int = 192,
        boundary_channels: int = 32,
        **_: object,
    ) -> None:
        super().__init__()
        if dino_img_size is None:
            dino_img_size = img_size
        if img_size % 16 != 0 or dino_img_size % 16 != 0:
            raise ValueError("DBANet input sizes must be divisible by 16")

        self.num_classes = int(num_classes)
                                            
        self.foundation_encoder = MultiDepthDINOv3SemanticEncoder(
            input_size=dino_img_size,
            num_prompts=num_prompts,
            pretrained=pretrained,
            layer_indices=self.DINO_LAYERS,
        )

                                         
        self.task_encoder = HierarchicalShapeAwareRWKVEncoder(
            image_size=img_size,
            pretrained=pretrained,
        )
        dims = self.task_encoder.DIMS

                                          
        self.foundation_fusion = AnchorPreservingMultiDepthFusion(
            self.foundation_encoder.embed_dim,
            neck_channels,
            len(self.DINO_LAYERS),
        )
        self.e3_proj = ConvNormAct(
            dims[3],
            neck_channels,
            kernel_size=1,
        )
        self.e4_proj = ConvNormAct(
            dims[4],
            neck_channels,
            kernel_size=1,
        )
                                        
        self.decoupling = (
            DINOv3GuidedBoundaryRegionDecouplingRefinement(
                neck_channels
            )
        )

                                         
                                            
        self.decoder_stage1 = MultiScaleBoundaryAwareDecoderStage(
            neck_channels,
            0,
            128,
            neck_channels,
            boundary_channels,
            num_classes,
        )
        self.decoder_stage2 = MultiScaleBoundaryAwareDecoderStage(
            128,
            dims[2],
            96,
            neck_channels,
            boundary_channels,
            num_classes,
        )
        self.decoder_stage3 = MultiScaleBoundaryAwareDecoderStage(
            96,
            dims[1],
            64,
            neck_channels,
            boundary_channels,
            num_classes,
        )
        self.decoder_stage4 = MultiScaleBoundaryAwareDecoderStage(
            64,
            dims[0],
            48,
            neck_channels,
            boundary_channels,
            num_classes,
        )
        self.segmentation_head = nn.Conv2d(
            48,
            num_classes,
            kernel_size=1,
        )
        self.register_buffer(
            "_prompt_init",
            self.foundation_encoder.prompts.detach().clone(),
        )
        self._mon: Dict[str, object] = {}

    def set_monitoring(self, enabled: bool = True) -> None:
                                        
        for module in self.modules():
            module._monitoring_enabled = bool(enabled)

    def get_param_groups(self) -> List[Dict[str, object]]:
                                          
        rwkv: List[nn.Parameter] = []
        prompts: List[nn.Parameter] = []
        new_layers: List[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name == "foundation_encoder.prompts":
                prompts.append(parameter)
            elif (
                name.startswith("task_encoder.encoder.")
                and ".offset_head." not in name
                and not name.endswith(".column_gate")
            ):
                rwkv.append(parameter)
            else:
                new_layers.append(parameter)
                                            
                                     
        return [
            {"params": rwkv, "lr_scale": 0.1},
            {"params": prompts, "lr_scale": 1.0},
            {"params": new_layers, "lr_scale": 1.0},
        ]

    @torch.no_grad()
    def monitor_stats(self) -> Dict[str, str]:
                                         
        out: Dict[str, str] = {}

        db = getattr(self.decoupling, "_mon", {})
        out["BRDR/foundation-task disagreement"] = (
            f"anchor-task dis={db.get('dis', 0):.3f} | "
            f"fuse|.|={db.get('fused', 0):.3f}"
        )
        out["BRDR/soft-frequency decoupling"] = (
            f"split={db.get('split', 0):.3f} | "
            f"hi-ratio={db.get('hi_ratio', 0):.3f} | "
            f"region|.|={db.get('region', 0):.3f} "
            f"boundary|.|={db.get('boundary', 0):.3f}"
        )
        out["BRDR/GR-KAN refinement"] = (
            f"effect={db.get('kan_effect', 0):.3f} | "
            f"backend={'GroupKANLinear' if _HAS_GKAN else 'Swish-MLP'}"
        )

        m = getattr(self, "_mon", {})
        out["BRDR/boundary prior"] = (
            f"|prior|={m.get('boundary_prior', 0):.3f} | "
            f"disagreement={m.get('disagreement', 0):.3f}"
        )

        shape_e1 = getattr(self.task_encoder.shape_e1, "_mon", {})
        shape_e2 = getattr(self.task_encoder.shape_e2, "_mon", {})
        out["HSRE/snake shape"] = (
            f"e1(off={shape_e1.get('off', 0):.4f}, delta={shape_e1.get('delta', 0):.3f}) "
            f"e2(off={shape_e2.get('off', 0):.4f}, delta={shape_e2.get('delta', 0):.3f})"
        )

        shifts = [getattr(wrapper, "_mon", {}) for wrapper in self.task_encoder.shift_wrappers()]
        out["HSRE/token-shift |off|"] = (
            "[" + ", ".join(f"{mon.get('off', 0):.4f}" for mon in shifts) + "]"
        )
        out["HSRE/column-gate tanh(g)"] = (
            "[" + ", ".join(f"{mon.get('col', 0):.3f}" for mon in shifts) + "]"
        )

        stages = [
            self.decoder_stage1,
            self.decoder_stage2,
            self.decoder_stage3,
            self.decoder_stage4,
        ]
        out["MBD/DySample |offset|"] = (
            "[" + ", ".join(f"{stage.up._mon.get('off', 0):.4f}" for stage in stages) + "]"
        )
        out["MBD/MSCA gamma"] = (
            "[" + ", ".join(f"{stage.context._mon.get('gamma', 0):.3f}" for stage in stages) + "]"
        )
        out["MBD/MSCA attention"] = (
            "[" + ", ".join(f"{stage.context._mon.get('attn', 0):.3f}" for stage in stages) + "]"
        )
        out["Heads/logit amplitude"] = (
            f"seg={m.get('seg_logit', 0):.3f} | "
            f"seg_deep={m.get('seg_deep', [])} | "
            f"boundary={m.get('boundary_logits', [])}"
        )

        prompts = self.foundation_encoder.prompts
        drift = float((prompts.detach() - self._prompt_init).norm())
        base = float(self._prompt_init.norm()) + 1e-8
        grad_norm = (
            float(prompts.grad.detach().norm())
            if prompts.grad is not None
            else float("nan")
        )
        fusion = self.foundation_fusion._mon
        weights = fusion.get("weights", [])
        out["MDSE/multi-depth fusion"] = (
            f"layers={list(self.DINO_LAYERS)} | "
            f"w={[round(value, 3) for value in weights]} | "
            f"gate={fusion.get('gate', 0):.3f} | "
            f"residual={fusion.get('residual_ratio', 0):.3f}"
        )
        out["MDSE/prompt tuning"] = (
            f"norm={float(prompts.detach().norm()):.3f} | "
            f"rel-drift={100.0 * drift / base:.1f}% | "
            f"grad-norm={grad_norm:.4f}"
        )
        return out

    def forward(
        self,
        x: Tensor,
        labels: Optional[Tensor] = None,
        return_aux: Optional[bool] = None,
    ) -> Union[Tensor, Dict[str, object]]:
\
\
\
\
\
           
        del labels
        if return_aux is None:
            return_aux = self.training
        input_size = x.shape[-2:]

                                            
        foundation_features = self.foundation_encoder(x)
        foundation, semantic_anchor = self.foundation_fusion(
            foundation_features
        )

                                                   
        e0, e1, e2, e3, e4 = self.task_encoder(
            x,
            return_shape_logits=return_aux,
        )
        task_e3 = self.e3_proj(e3)
        task_e4 = self.e4_proj(e4)

                                                 
                                        
        target_size = task_e4.shape[-2:]
        foundation = F.interpolate(
            foundation,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        semantic_anchor = F.interpolate(
            semantic_anchor,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        task_e3 = F.interpolate(
            task_e3,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        region, boundary, disagreement = self.decoupling(
            foundation,
            semantic_anchor,
            task_e3,
            task_e4,
        )

                                              
        stage1, seg_stage1, boundary_stage1 = self.decoder_stage1(
            region,
            None,
            boundary,
            return_aux=return_aux,
        )
        stage2, seg_stage2, boundary_stage2 = (
            self.decoder_stage2(
                stage1,
                e2,
                boundary,
                return_aux=return_aux,
            )
        )
        stage3, seg_stage3, boundary_stage3 = self.decoder_stage3(
            stage2,
            e1,
            boundary,
            return_aux=return_aux,
        )
        stage4, seg_stage4, final_boundary = self.decoder_stage4(
            stage3,
            e0,
            boundary,
            return_aux=return_aux,
        )
        segmentation = self.segmentation_head(stage4)
        if segmentation.shape[-2:] != input_size:
            segmentation = F.interpolate(
                segmentation,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        if _monitoring_enabled(self) and return_aux:
            with torch.no_grad():
                self._mon = {
                    "boundary_prior": float(boundary.detach().abs().mean()),
                    "disagreement": float(disagreement.detach().mean()),
                    "seg_logit": float(segmentation.detach().abs().mean()),
                    "seg_deep": [
                        round(float(t.detach().abs().mean()), 3)
                        for t in (seg_stage4, seg_stage3, seg_stage2, seg_stage1)
                    ],
                    "boundary_logits": [
                        round(float(t.detach().abs().mean()), 3)
                        for t in (
                            final_boundary,
                            boundary_stage3,
                            boundary_stage2,
                            boundary_stage1,
                        )
                    ],
                }
        if not return_aux:
            return segmentation

        return {
            "seg": segmentation,
            "seg_deep": [
                self._resize_logits(seg_stage4, input_size),
                self._resize_logits(seg_stage3, input_size),
                self._resize_logits(seg_stage2, input_size),
                self._resize_logits(seg_stage1, input_size),
            ],
            "boundary": [
                self._resize_logits(final_boundary, input_size),
                self._resize_logits(boundary_stage3, input_size),
                self._resize_logits(boundary_stage2, input_size),
                self._resize_logits(boundary_stage1, input_size),
            ],
                                                
            "snake_curve": list(self.task_encoder._snake_curve),
            "disagreement": disagreement,
        }

    @staticmethod
    def _resize_logits(logits: Tensor, size: Tuple[int, int]) -> Tensor:
                                            
        if logits.shape[-2:] == size:
            return logits
        return F.interpolate(
            logits,
            size=size,
            mode="bilinear",
            align_corners=False,
        )


__all__ = [
    "DBANet",
    "MultiDepthDINOv3SemanticEncoder",
    "AnchorPreservingMultiDepthFusion",
    "HierarchicalShapeAwareRWKVEncoder",
    "DINOv3GuidedBoundaryRegionDecouplingRefinement",
    "MultiScaleBoundaryAwareDecoderStage",
    "FoundationSemanticEncoder",
    "GRKANBlock",
    "SnakeShapeBranch",
    "DeformableTokenShiftWKV",
    "ShapeAwareRWKVEncoder",
    "SoftFrequencyDecoupling",
    "FoundationGuidedDBRD",
    "DynamicSampleUpsampler",
    "MultiScaleStripContext",
    "MBDStage",
]


def _checkpoint_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state_dict", "state_dict", "model", "net"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def main():
    parser = argparse.ArgumentParser(prog="DBANet")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--aux", action="store_true")
    args = parser.parse_args()
    image_size = args.image_size or (512 if args.device.startswith("cuda") else 64)
    model = DBANet(
        num_classes=args.num_classes,
        img_size=image_size,
        pretrained=args.pretrained,
    )
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = _checkpoint_state(checkpoint)
        model.load_state_dict(state, strict=True)
    model = model.to(args.device).eval()
    model.set_monitoring(False)
    sample = torch.randn(1, 3, image_size, image_size, device=args.device)
    with torch.inference_mode():
        output = model(sample, return_aux=args.aux)
    if isinstance(output, dict):
        shapes = {
            "seg": tuple(output["seg"].shape),
            "seg_deep": [tuple(value.shape) for value in output["seg_deep"]],
            "boundary": [tuple(value.shape) for value in output["boundary"]],
            "snake_curve": [tuple(value.shape) for value in output["snake_curve"]],
            "disagreement": tuple(output["disagreement"].shape),
        }
    else:
        shapes = tuple(output.shape)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print({"output": shapes, "total_parameters": total, "trainable_parameters": trainable})


if __name__ == "__main__":
    main()
