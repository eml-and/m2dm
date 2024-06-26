import math
import os
import sys
from abc import abstractmethod

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from improved_diff.fp16_util import convert_module_to_f16, convert_module_to_f32
from improved_diff.nn import (
    SiLU,
    avg_pool_nd,
    checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)
from improved_diff.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from improved_diff.train_util import TrainLoop

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from src.image_train import create_argparser


def blockdiag_matmul(x, w):
    """
    x is expected to be of shape [..., sqrt_n * sqrt_n]
    w is expected to be of shape [sqrt_n, sqrt_n, sqrt_n]
    Reshape x to [..., sqrt_n, sqrt_n]
    reshaped_x = x.view(*x.shape[:-1], w.shape[0], w.shape[-1])
    Perform the block diagonal matrix multiplication
    result = th.einsum("bnm,...bm->...bn", w, reshaped_x)
    Reshape the result back to the original shape of x
    return result.reshape(*x.shape)
    """
    return th.einsum(
        "bnm,...bm ->... bn", w, x.view(*x.shape[:-1], w.shape[0], w.shape[-1])
    ).reshape(*x.shape)


class MonarchMatrix(nn.Module):

    def __init__(self, sqrt_n: int):
        super().__init__()
        self.sqrt_n = sqrt_n
        self.L = nn.Parameter(th.randn((sqrt_n, sqrt_n, sqrt_n)))
        self.R = nn.Parameter(th.randn((sqrt_n, sqrt_n, sqrt_n)))

    def forward(self, x):
        x = rearrange(x, "... (m n) -> ... (n m)", n=self.sqrt_n)
        x = blockdiag_matmul(x, self.L)
        x = rearrange(x, "... (m n) -> ... (n m)", n=self.sqrt_n)
        x = blockdiag_matmul(x, self.R)
        return rearrange(x, " ... (m n) -> ... (n m)", n=self.sqrt_n)


class MonarchMixerLayer(nn.Module):
    """
    Eq. 2 and Eq. 3 from the M2 paper,
    where
    Eq. 2: convolution wit kernel
    Eq. 3: MLP with monarch matrices
    Processes input sequences of embeddings (b,n,d)
    """

    def __init__(self, sqrt_n: int, sqrt_d: int):
        super().__init__()
        self.m1 = MonarchMatrix(sqrt_n)
        self.m2 = MonarchMatrix(sqrt_n)
        self.m3 = MonarchMatrix(sqrt_d)
        self.m4 = MonarchMatrix(sqrt_d)

        self.n_kernel = nn.Parameter(th.randn(sqrt_d**2, sqrt_n**2))
        self.d_kernel = nn.Parameter(th.randn(1, sqrt_d**2))
        self.layer_norm = nn.LayerNorm(sqrt_d**2)

    def forward(self, x: th.Tensor):  # x. shape = (b, n, d)
        """
        x_tilde: mix along sequence axis (Convolution)
        y: mix along embedding axis (MLP)
        """
        x_tilde = self.m2(
            th.relu(self.n_kernel * self.m1(x.transpose(-1, -2)))
        ).transpose(
            -1, -2
        )  # mix sequence
        y = self.m4(th.relu(self.d_kernel * self.m3(x_tilde)))  # mix features
        return self.layer_norm(y + x_tilde)  # skip connection


class MonarchConv(nn.Module):
    """
    Eq. 2 from the M2 paper (conv)
    >> Mix along sequence
    """

    def __init__(self, sqrt_n: int, sqrt_d: int):
        super().__init__()
        self.m1 = MonarchMatrix(sqrt_n)
        self.m2 = MonarchMatrix(sqrt_n)
        self.n_kernel = nn.Parameter(th.randn(sqrt_d**2, sqrt_n**2))

    def forward(self, x: th.Tensor):
        x_tilde = self.m2(
            th.relu(self.n_kernel * self.m1(x.transpose(-1, -2)))
        ).transpose(-1, -2)
        return x_tilde


class MonarchLinear(nn.Module):
    def __init__(self, sqrt_d: int):
        super().__init__()
        self.m3 = MonarchMatrix(sqrt_d)
        self.m4 = MonarchMatrix(sqrt_d)

        self.layer_norm = nn.LayerNorm(sqrt_d**2)

    def forward(self, x):
        output = self.m4(self.m3(x))
        return self.layer_norm(output)


class MonarchMLP(nn.Module):
    """
    Eq. 3 from the M2 paper (MLP)
    >> Mix along features
    """

    def __init__(self, sqrt_d: int):
        super().__init__()
        self.m3 = MonarchMatrix(sqrt_d)
        self.m4 = MonarchMatrix(sqrt_d)

    def forward(self, x_tilde: th.Tensor):
        y = self.m4(th.relu(self.m3(x_tilde.transpose(-1, -2))))  # mix features
        return y.transpose(-1, -2)


class MonarchDimMix(nn.Module):
    """
    Linear-replacer (MLP) in BERT for mixing along the dimension axis
    """

    def __init__(self, sqrt_d: int):
        super().__init__()
        self.m3 = MonarchMatrix(sqrt_d)
        self.m4 = MonarchMatrix(sqrt_d)
        self.m5 = MonarchMatrix(sqrt_d)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, x):
        x_1 = self.sigmpoid(self.m3(x))
        x_2 = self.m4(x)
        return self.m5(self.GELU(x_1 * x_2))


class GatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(GatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gate_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv_out = self.conv(x)
        gate_out = self.sigmoid(self.gate_conv(x))
        return conv_out * gate_out


class MonarchGatedConv(nn.Module):
    """
    alternatives to Eq. 2 (conv) from M2 paper
    V matmul M2 (K1 matmul M1 (Q matmul K)) where Q,K,V are
    linear projections of X, reproduces a gated convolution block

    TODO: ablation if MonarchLinear or MonarchMLP works better
    TODO: ablation of LayerNorm is necessary
    """

    def __init__(self, sqrt_n: int, sqrt_d: int, res: bool):
        super().__init__()
        self.res = res
        self.q = MonarchLinear(sqrt_d=sqrt_d)
        self.k = MonarchLinear(sqrt_d=sqrt_d)
        self.v = MonarchLinear(sqrt_d=sqrt_d)

        self.m2, self.m1 = MonarchMatrix(sqrt_n), MonarchMatrix(sqrt_n)
        self.m9, self.m8 = MonarchMatrix(sqrt_n), MonarchMatrix(sqrt_n)

        self.d_kernel = nn.Parameter(th.randn(1, sqrt_d**2))
        self.d_kernel_bi = nn.Parameter(th.randn(1, sqrt_d**2))

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        h = V * self.m2(self.d_kernel * self.m1(Q * K))
        if not self.res:
            return h
        else:
            x_res = self.m9(self.d_kernel * self.m8(x))
        return x_res + h


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        TODO: use ths to compare the number of operations from QKV and QKV free models
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial**2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class M2UNetModel(nn.Module):
    """
    The full M2UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    # device = "cuda"
    # # Namespace
    # args = create_argparser().parse_args()

    # model, diffusion = create_model_and_diffusion(
    #     **args_to_dict(args, model_and_diffusion_defaults().keys())
    # )
    # model.to(device)
    x = th.randn(2, 3, 36, 36)
    ### Matrix Test
    # mm = MonarchMatrix(sqrt_n=8)

    # x_h = mm.forward(x)
    # print(x_h.shape)

    ### Model Test
    sqrt_n = 6
    sqrt_d = 6
    model = MonarchMixerLayer(sqrt_n, sqrt_d)
    out = model(x)
    breakpoint()

    model_ResBlock = nn.Sequential(
        ResBlock(
            channels=64,
            emb_channels=4 * 64,
            out_channels=1 * 64,
            use_conv=False,
            dims=2,
            dropout=0,
        ),
        ResBlock(
            channels=64,
            emb_channels=4 * 64,
            out_channels=1 * 64,
            use_conv=False,
            dims=2,
            dropout=0,
        ),
        ResBlock(
            channels=128,
            emb_channels=4 * 128,
            out_channels=1 * 128,
            use_conv=False,
            dims=2,
            dropout=0,
        ),
    )

    model_MML = nn.Sequential(
        MonarchMixerLayer(6, 6),
        MonarchMixerLayer(5, 5),
        MonarchMixerLayer(4, 4),
    )
    print(count_parameters(model_ResBlock))  # 542464
    print(count_parameters(model_MML))


if __name__ == "__main__":
    main()
