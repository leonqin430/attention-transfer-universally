# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Attention Transfer: https://github.com/alexlioralexli/attention-transfer
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# DINOv2 (LayerScale): https://github.com/facebookresearch/dinov2
# CLIP (pre-LayerNorm): https://github.com/openai/CLIP
# BEiTv2 (relative position bias): https://github.com/microsoft/unilm/tree/master/beit2
# --------------------------------------------------------

"""Teacher-student Vision Transformer for Attention Transfer.

This file extends the DualVisionTransformer of Li et al. (2024) with three optional
architectural components, so that a student can be instantiated with the teacher's
*native* architecture (the native-architecture rescue of our paper):

  * LayerScale            (DINOv2, DINOv2-wr, BEiTv2)  -> parameters blocks.{i}.gamma1 / gamma2
  * pre-LayerNorm         (CLIP)                       -> module ln_pre
  * RelativePositionBias  (BEiTv2)                     -> parameter blocks.{i}.attn.relative_position_bias_table

Two transfer modes are supported, exactly as in Li et al.:

  * 'copy'    : Attention Copy. The student's attention maps are replaced by the teacher's.
  * 'distill' : Attention Distillation. The student learns its own attention, with a
                cross-entropy loss between the teacher's and the student's per-layer
                attention maps (summed over the transferred layers).

The transferred quantity is always the raw query-key attention logits (q @ k^T * scale),
i.e. the relative position bias of a BEiTv2 model is applied inside its own forward pass
but is not part of the transferred attention.
"""
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import trunc_normal_


def make_drop_path_masks(dpr, batch_size, dtype, device, training):
    """Per-sample stochastic-depth masks, one (attention, MLP) pair per block.

    The same masks are applied to the teacher and the student of a DualVisionTransformer
    so that both streams see identical stochastic depth (as in Li et al.).
    """
    if not training:
        return [None] * len(dpr)
    drop_masks = []
    shape = (batch_size,)
    for rate in dpr:
        curr_layer_masks = []
        for _ in range(2):
            keep_prob = 1 - rate
            random_tensor = keep_prob + torch.rand(shape, dtype=dtype, device=device)
            random_tensor.floor_()  # binarize
            curr_layer_masks.append(random_tensor / keep_prob)
        drop_masks.append(curr_layer_masks)
    return drop_masks


def attention_cross_entropy(teacher_logits, student_logits):
    """Attention Distillation loss of Li et al.: CE(softmax(teacher), softmax(student))
    computed per query over the key dimension, averaged over batch, heads and queries."""
    return -(teacher_logits.softmax(dim=-1) * torch.log_softmax(student_logits, dim=-1)).sum(dim=-1).mean()


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, mode=None,
                 use_rpb=False, num_tokens=197):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.mode = mode

        # Relative position bias (BEiTv2-style) over a (sqrt(N-1) x sqrt(N-1)) patch grid plus CLS.
        if use_rpb:
            window_size = int((num_tokens - 1) ** 0.5)  # 14 for 197 tokens
            num_relative_distance = (2 * window_size - 1) * (2 * window_size - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(num_relative_distance, num_heads))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, H, W
            coords_flatten = torch.flatten(coords, 1)  # 2, H*W
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
            relative_coords[:, :, 0] += window_size - 1
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # N_patch, N_patch

            rpi = torch.zeros(num_tokens, num_tokens, dtype=torch.long)
            rpi[1:, 1:] = relative_position_index
            rpi[0, 0] = num_relative_distance - 1    # CLS-to-CLS
            rpi[0, 1:] = num_relative_distance - 2   # CLS-to-patch
            rpi[1:, 0] = num_relative_distance - 3   # patch-to-CLS
            self.register_buffer("relative_position_index", rpi)
        else:
            self.relative_position_bias_table = None
            self.relative_position_index = None

    def _get_rel_pos_bias(self):
        if self.relative_position_bias_table is None:
            return None
        return self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.relative_position_index.shape[0],
               self.relative_position_index.shape[1], -1).permute(2, 0, 1)  # nH, N, N

    def forward(self, x, teacher_act=None, return_act=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale  # raw QK logits, without any position bias

        if self.mode == 'copy' and teacher_act is not None:
            # Attention Copy: this layer uses the teacher's attention map
            attn = teacher_act.softmax(dim=-1)
        else:
            logits = attn_logits
            rel_pos_bias = self._get_rel_pos_bias()
            if rel_pos_bias is not None:
                logits = logits + rel_pos_bias.unsqueeze(0)
            attn = logits.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if return_act == 'attention':
            return x, attn_logits
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, mode=None,
                 use_layerscale=False, layerscale_init=1e-5,
                 use_rpb=False, num_tokens=197):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, mode=mode,
            use_rpb=use_rpb, num_tokens=num_tokens)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=0)

        # LayerScale (CaiT / DINOv2 / BEiTv2-style)
        if use_layerscale:
            self.gamma1 = nn.Parameter(layerscale_init * torch.ones(dim))
            self.gamma2 = nn.Parameter(layerscale_init * torch.ones(dim))
        else:
            self.gamma1 = None
            self.gamma2 = None

    def forward(self, x, drop_masks=None, teacher_act=None, return_act=None):
        if drop_masks is None:
            drop_masks = (1, 1)
        else:
            shape = (len(x),) + (1,) * (x.ndim - 1)
            drop_masks = (drop_masks[0].view(*shape), drop_masks[1].view(*shape))
        attn_outputs = self.attn(self.norm1(x), teacher_act=teacher_act, return_act=return_act)
        if return_act is not None:
            attn_result, act = attn_outputs
        else:
            attn_result = attn_outputs

        if self.gamma1 is not None:
            x = x + drop_masks[0] * self.gamma1 * attn_result
            x = x + drop_masks[1] * self.gamma2 * self.mlp(self.norm2(x))
        else:
            x = x + drop_masks[0] * attn_result
            x = x + drop_masks[1] * self.mlp(self.norm2(x))
        if return_act is not None:
            return x, act
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer with optional LayerScale, pre-LayerNorm and relative position bias.

    Used as the teacher and the student stream of DualVisionTransformer, and as a standalone
    model for the native-architecture No-Transfer baselines (see models_vit.py).
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 norm_layer=nn.LayerNorm, mode=None, global_pool=False, drop_path_rate=0.,
                 use_layerscale=False, layerscale_init=1e-5,
                 use_ln_pre=False, use_rpb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.mode = mode

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        num_tokens = num_patches + 1  # +1 for CLS

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # CLIP-style pre-LayerNorm applied to the token sequence before the first block
        self.ln_pre = norm_layer(embed_dim) if use_ln_pre else None

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, norm_layer=norm_layer, mode=mode,
                use_layerscale=use_layerscale, layerscale_init=layerscale_init,
                use_rpb=use_rpb, num_tokens=num_tokens)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # stochastic depth, only used when the model is run standalone (inside a
        # DualVisionTransformer the wrapper provides the masks)
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, drop_masks=None, teacher_act=None, return_act=None):
        if teacher_act is not None:
            assert self.mode == 'copy'
            # pad if we copy fewer blocks
            if len(teacher_act) < len(self.blocks):
                teacher_act = teacher_act + [None] * (len(self.blocks) - len(teacher_act))
        else:
            teacher_act = [None] * len(self.blocks)

        B = x.shape[0]
        if drop_masks is None:  # standalone use
            drop_masks = make_drop_path_masks(self.dpr, B, x.dtype, x.device, self.training)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        if self.ln_pre is not None:
            x = self.ln_pre(x)

        attns = []
        for i, blk in enumerate(self.blocks):
            if return_act is not None:
                x, act = blk(x, return_act=return_act, drop_masks=drop_masks[i])
                attns.append(act)
            else:
                x = blk(x, teacher_act=teacher_act[i], drop_masks=drop_masks[i])

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        if return_act is not None:
            return outcome, attns
        return outcome

    def forward(self, x, drop_masks=None, teacher_act=None, return_act=None):
        x = self.forward_features(x, drop_masks=drop_masks, teacher_act=teacher_act, return_act=return_act)
        if return_act is not None:
            x, act = x
            return self.head(x), act
        else:
            return self.head(x)


class DualVisionTransformer(nn.Module):
    """
    Two ViT streams: a frozen pre-trained teacher and a randomly initialized student.

    mode='copy'    -> the student uses the teacher's attention maps in the transferred layers
    mode='distill' -> training returns (student_logits, attention distillation loss)
    end_layer (<= 0) -> number of final layers that are NOT transferred (0 = transfer all layers)
    """
    def __init__(self, mode='distill', drop_path_rate=0,
                 teacher_kwargs=None, student_kwargs=None, end_layer=0):
        super().__init__()
        assert mode in {'copy', 'distill'}, f"unsupported mode {mode!r}"
        assert end_layer <= 0, "end_layer is a non-positive offset from the last layer"
        self.mode = mode
        self.drop_path_rate = drop_path_rate
        self.teacher_depth = teacher_kwargs['depth']
        self.student_depth = student_kwargs['depth']
        assert self.teacher_depth == self.student_depth
        self.teacher = VisionTransformer(mode='teacher', **teacher_kwargs)
        self.student = VisionTransformer(mode=mode, **student_kwargs)
        self.dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.teacher_depth)]
        self.end_layer = end_layer
        self.n_transfer = self.teacher_depth + self.end_layer  # layers [0, n_transfer) are transferred

    def forward(self, x):
        drop_masks = make_drop_path_masks(self.dpr, len(x), x.dtype, x.device, self.training)

        teacher_act = None
        with torch.no_grad():
            if self.training or self.mode == 'copy':
                _, teacher_act = self.teacher.forward_features(x, drop_masks=drop_masks,
                                                               return_act='attention')
                teacher_act = [act.detach() for act in teacher_act]

        if self.mode == 'copy':
            return self.student(x, drop_masks=drop_masks, teacher_act=teacher_act[:self.n_transfer])

        if self.training:  # distill
            student_out, student_act = self.student(x, drop_masks=drop_masks, return_act='attention')
            distill_loss = 0
            for i in range(self.n_transfer):
                distill_loss = distill_loss + attention_cross_entropy(teacher_act[i], student_act[i])
            return student_out, distill_loss

        return self.student(x, drop_masks=drop_masks)

    def no_weight_decay(self):
        return {'student.' + k for k in self.student.no_weight_decay()}


# ------------------------------------------------------------------------------------
# Model factories. The teacher and the student always share the same architecture.
# ------------------------------------------------------------------------------------

_NORM = partial(nn.LayerNorm, eps=1e-6)

_TINY = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, norm_layer=_NORM)
_SMALL = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=_NORM)
_BASE = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=_NORM)
_LARGE = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=_NORM)


def _dual(arch, mode, drop_path_rate, end_layer, native=None, **kwargs):
    vit_kwargs = dict(arch, **(native or {}), **kwargs)  # teacher and student share the architecture
    return DualVisionTransformer(mode=mode, drop_path_rate=drop_path_rate, end_layer=end_layer,
                                 teacher_kwargs=vit_kwargs, student_kwargs=vit_kwargs)


# ---- standard teacher-student architecture (vanilla ViT, as in Li et al.) ----

def dual_vit_tiny_patch16(mode='distill', drop_path_rate=0, end_layer=0, **kwargs):
    return _dual(_TINY, mode, drop_path_rate, end_layer, **kwargs)


def dual_vit_small_patch16(mode='distill', drop_path_rate=0, end_layer=0, **kwargs):
    return _dual(_SMALL, mode, drop_path_rate, end_layer, **kwargs)


def dual_vit_base_patch16(mode='distill', drop_path_rate=0, end_layer=0, **kwargs):
    return _dual(_BASE, mode, drop_path_rate, end_layer, **kwargs)


def dual_vit_large_patch16(mode='distill', drop_path_rate=0, end_layer=0, **kwargs):
    return _dual(_LARGE, mode, drop_path_rate, end_layer, **kwargs)


# ---- native-architecture teacher-student pairs (native-architecture rescue) ----

LAYERSCALE = dict(use_layerscale=True, layerscale_init=1e-5)
LN_PRE = dict(use_ln_pre=True)
RPB = dict(use_rpb=True)


def dual_vit_dinov2_small_patch16(mode='distill', drop_path_rate=0, end_layer=0, **kwargs):
    """DINOv2-S / DINOv2-wr-S native architecture: ViT-Small with LayerScale."""
    return _dual(_SMALL, mode, drop_path_rate, end_layer, native=LAYERSCALE, **kwargs)


def dual_vit_dinov2_base_patch16(mode='distill', drop_path_rate=0, end_layer=0, **kwargs):
    """DINOv2-B / DINOv2-wr-B native architecture: ViT-Base with LayerScale."""
    return _dual(_BASE, mode, drop_path_rate, end_layer, native=LAYERSCALE, **kwargs)


def dual_vit_clip_base_patch16(mode='distill', drop_path_rate=0, end_layer=0, **kwargs):
    """CLIP-B native architecture: ViT-Base with pre-LayerNorm (ln_pre)."""
    return _dual(_BASE, mode, drop_path_rate, end_layer, native=LN_PRE, **kwargs)


def dual_vit_beitv2_base_patch16(mode='distill', drop_path_rate=0, end_layer=0, **kwargs):
    """BEiTv2-B native architecture: ViT-Base with LayerScale and relative position bias."""
    return _dual(_BASE, mode, drop_path_rate, end_layer, native={**LAYERSCALE, **RPB}, **kwargs)
