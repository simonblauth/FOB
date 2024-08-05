"""
Adjusted Vision Transformer with μP-scaling from https://arxiv.org/abs/2203.03466
"""

from torch import nn
from timm.models.vision_transformer import VisionTransformer, named_apply
from mup import MuReadout
from mup.init import trunc_normal_, normal_


class WidthScalingVisionTransformer(VisionTransformer):
    def __init__(self, *args, width: int=12, replace_head: bool=True, **kwargs):
        embed_dim_per_head = 64
        kwargs["embed_dim"] = width * embed_dim_per_head
        kwargs["num_heads"] = width
        if "patch_size" not in kwargs:
            kwargs["patch_size"] = 4
        if "img_size" not in kwargs:
            kwargs["img_size"] = 64
        if "num_classes" not in kwargs:
            kwargs["num_classes"] = 1000
        super().__init__(*args, **kwargs)
        if replace_head:
            self.head = MuReadout(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()

    def reset_weights(self, mode: str = ''):
        """Override custom timm initialization with μP-scaling init. Must be called after calling `set_base_shapes`."""
        assert mode == '', "Only supports timm init for now"
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            normal_(self.cls_token, std=1e-6)
        named_apply(init_weights_vit_timm, self)

def init_weights_vit_timm(module: nn.Module, name: str = ''):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    # Note that only custom inits are overridden. Regular torch init is handled by `set_base_shapes`.
