
# !!!This repository currently provides part of the code, including the model implementation and backbone initialization weights. 
# !!!The complete code, including data preprocessing, training, and evaluation pipelines, will be released upon acceptance of the manuscript.


from __future__ import annotations

import argparse
from pathlib import Path
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


def _local_weight_path(filename: str, pretrained_dir=None) -> Optional[Path]:
    directory = Path(pretrained_dir) if pretrained_dir is not None else Path(__file__).resolve().parent / "weights"
    path = directory / filename
    if path.is_file():
        return path
    if pretrained_dir is not None:
        raise FileNotFoundError(f"Required pretrained backbone weights not found: {path}")
    return None


def _create_dino_backbone(model_name: str, input_size: int, pretrained: bool, pretrained_dir=None):
    path = _local_weight_path("dinov3_vitl16_sat493m.safetensors", pretrained_dir) if pretrained else None
    if path is not None and model_name != "vit_large_patch16_dinov3.sat493m":
        raise ValueError("The bundled DINOv3 checkpoint is for vit_large_patch16_dinov3.sat493m")
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained and path is None,
        in_chans=3,
        num_classes=0,
        img_size=input_size,
    )
    if path is not None:
        from safetensors.torch import load_file

        backbone.load_state_dict(load_file(str(path), device="cpu"), strict=True)
        print(f"[DBANet/MDSE] DINOv3 pretrained weights loaded: {path}")
    return backbone





MBDStage = MultiScaleBoundaryAwareDecoderStage


class DBANet(nn.Module):


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
        pretrained_dir: Optional[str] = None,
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
            pretrained_dir=pretrained_dir,
        )


        self.task_encoder = HierarchicalShapeAwareRWKVEncoder(
            image_size=img_size,
            pretrained=pretrained,
            pretrained_dir=pretrained_dir,
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
    parser.add_argument("--pretrained-dir", help="Directory containing the two backbone initialization checkpoints; implies --pretrained")
    parser.add_argument("--aux", action="store_true")
    args = parser.parse_args()
    image_size = args.image_size or (512 if args.device.startswith("cuda") else 64)
    model = DBANet(
        num_classes=args.num_classes,
        img_size=image_size,
        pretrained=args.pretrained or args.pretrained_dir is not None,
        pretrained_dir=args.pretrained_dir,
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
