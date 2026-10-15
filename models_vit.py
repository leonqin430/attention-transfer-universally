# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Attention Transfer: https://github.com/alexlioralexli/attention-transfer
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

"""Single-stream Vision Transformers, used for the No-Transfer (NoT) baselines.

* vit_{tiny,small,base,large}_patch16 : the standard ViT of Li et al. (timm-based), for the
  standard-architecture No-Transfer baseline.
* vit_dinov2_*, vit_clip_*, vit_beitv2_* : the native architectures (LayerScale, pre-LayerNorm,
  relative position bias) for the native-architecture No-Transfer baselines. They share the
  implementation of the teacher/student streams in models_dual_vit.py.
"""

from functools import partial

import torch
import torch.nn as nn

import timm
import timm.models.vision_transformer

import models_dual_vit


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x, layer_to_return=None):
        # return features up until layer_to_return
        intermediate_feat = []
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if layer_to_return is not None and i < layer_to_return:
                intermediate_feat.append(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        if layer_to_return is not None:
            return outcome, intermediate_feat
        else:
            return outcome

    def forward(self, x, return_features=False, layer_to_return=None):
        if layer_to_return is not None:
            final_feats, intermediate_feats = self.forward_features(x, layer_to_return=layer_to_return)
        else:
            final_feats = self.forward_features(x)
        pred = self.head(final_feats)
        if return_features and layer_to_return is None:
            return pred, final_feats
        elif return_features and layer_to_return is not None:
            return pred, intermediate_feats
        else:
            return pred


def vit_tiny_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# ------------------------------------------------------------------------------------
# Native-architecture single-stream models (native-architecture No-Transfer baselines)
# ------------------------------------------------------------------------------------

def _native_vit(arch, **kwargs):
    return models_dual_vit.VisionTransformer(**dict(arch, **kwargs))


def vit_dinov2_small_patch16(**kwargs):
    """DINOv2-S native architecture: ViT-Small with LayerScale."""
    return _native_vit(models_dual_vit._SMALL, **models_dual_vit.LAYERSCALE, **kwargs)


def vit_dinov2_base_patch16(**kwargs):
    """DINOv2-B native architecture: ViT-Base with LayerScale."""
    return _native_vit(models_dual_vit._BASE, **models_dual_vit.LAYERSCALE, **kwargs)


def vit_clip_base_patch16(**kwargs):
    """CLIP-B native architecture: ViT-Base with pre-LayerNorm."""
    return _native_vit(models_dual_vit._BASE, **models_dual_vit.LN_PRE, **kwargs)


def vit_beitv2_base_patch16(**kwargs):
    """BEiTv2-B native architecture: ViT-Base with LayerScale and relative position bias."""
    return _native_vit(models_dual_vit._BASE, **models_dual_vit.LAYERSCALE, **models_dual_vit.RPB, **kwargs)
