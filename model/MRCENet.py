## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
from torchvision.transforms import *
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math

from einops import rearrange, repeat
from .torch_wavelets import DWT_2D, IDWT_2D
import numpy as np
from matplotlib import pyplot as plt

## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class LayerNorm2d(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SimpleGte(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        #
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.conv = nn.Sequential(  # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features),
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv(x1)
        x2 = F.relu(x2)
        x = x1 * x2
        x = self.project_out(x)
        return x
    
def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    # subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    # compute exponentials
    exp_x = torch.exp(x)
    # compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

##########################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, shift_size, window_size):
        super(Attention, self).__init__()
        self.dim = dim
        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.bias = bias
        self.shift_size = shift_size
        self.window_size = window_size

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([
            relative_coords_h,
            relative_coords_w], indexing="ij")).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        relative_coords_table[:, :, :, 0] /= (self.window_size - 1)
        relative_coords_table[:, :, :, 1] /= (self.window_size - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / math.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table, persistent=False)
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def window_partition(self, x, window_size):
        """
        Args:
            x: (b,c, h, w, )
            window_size (int): window size

        Returns:
            windows: (num_windows*b, c, window_size, window_size,)
        """
        b, c, h, w, = x.shape
        x = x.view(b, c, h // window_size, window_size, w // window_size, window_size, )
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, c, window_size, window_size, )
        return windows

    def calculate_mask(self, h, w):
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, 1, h, w,))  # 1 1 h w
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w, ] = cnt
                cnt += 1

        mask_windows = self.window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    @torch.jit.ignore
    def no_weight_decay(self):
        nod = set()
        for n, m in self.named_modules():
            if any([kw in n for kw in ("cpb_mlp", "logit_scale", 'relative_position_bias_table')]):
                nod.add(n)
        return nod

    def window_reverse(self, windows, window_size, h, w):
        """
        Args:
            windows: (num_windows*b,c, window_size, window_size, )
            window_size (int): Window size
            h (int): Height of image
            w (int): Width of image

        Returns:
            x: (b,c, h, w, )
        """
        b = int(windows.shape[0] / (h * w / window_size / window_size))
        x = windows.view(b, h // window_size, w // window_size, -1, window_size, window_size, )
        x = x.permute(0, 3, 1, 4, 2, 5, ).contiguous().view(b, -1, h, w, )
        return x

    def window_attn(self, qkv, h, w, wb, wh, ww):
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head (h w) c ', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        if self.shift_size > 0:
            mask = self.calculate_mask(h, w)
        else:
            mask = None
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(wb // nw, nw, self.num_heads, wh * ww, wh * ww) + mask.unsqueeze(1).unsqueeze(0).to(
                attn.device)
            attn = attn.view(-1, self.num_heads, wh * ww, wh * ww)

        attn = softmax_one(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=wh, w=ww)

        return out

    def channel_attn(self, qkv, h, w):
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads).contiguous(
            memory_format=torch.contiguous_format)

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1) 
        # attn = attn.softmax(dim=-1)
        attn = attn * self.temperature
        attn = softmax_one(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return out

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv_window, qkv_ch = torch.split(qkv, c * 3 // 2, dim=1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(qkv_window, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = qkv_window

        # partition windows
        x = self.window_partition(shifted_x, self.window_size)  # nw*b,c, window_size, window_size,
        # window shape
        wb, wc, wh, ww = x.shape

        x_win = self.window_attn(x, h, w, wb, wh, ww)

        # shifted_x = self.project_out(x_win)
        shifted_x = self.window_reverse(x_win, self.window_size, h, w)  # b c h w

        # reverse cyclic shift
        if self.shift_size > 0:
            out_win = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            out_win = shifted_x

        x_channel = self.channel_attn(qkv_ch, h, w)

        out = self.project_out(torch.cat([out_win, x_channel], dim=1))

        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, shift_size, window_size):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm2d(dim, data_format="channels_first")
        self.attn = Attention(dim, num_heads=num_heads, bias=bias, shift_size=shift_size, window_size=window_size)
        self.norm2 = LayerNorm2d(dim, data_format="channels_first")
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.PA = PA(embed_dim)
        #self.down = nn.PixelUnshuffle(2)

    def forward(self, x):
        x = self.proj(x)
        #x = self.PA(x)
        #x = self.down(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat, out_dim):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                 )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

    def forward(self, x):
        return self.body(x)

class Simple_spatial_attention(nn.Module):
    def __init__(self, in_dim):
        super(Simple_spatial_attention, self).__init__()
        self.conv = nn.Conv2d(in_dim, 1, kernel_size=1, padding=0, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.conv(x))


class Simple_channel_attention(nn.Module):
    def __init__(self, in_dim):
        super(Simple_channel_attention, self).__init__()
        self.conv = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=True))

    def forward(self, x):
        return x * self.conv(x)


class ASGFM(nn.Module):
    def __init__(self, mri_dim, pet_dim, hidden_dim, num_tokens=64):
        super().__init__()
        self.num_tokens = num_tokens

        self.norm1 = LayerNorm2d(hidden_dim, data_format="channels_first")
        self.norm2 = LayerNorm2d(hidden_dim, data_format="channels_first")

        self.q1 = nn.Sequential(nn.Conv2d(mri_dim, hidden_dim, kernel_size=1),
                                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
                                )
        self.q2 = nn.Sequential(nn.Conv2d(pet_dim, hidden_dim, kernel_size=1),
                                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
                                )

        self.conv1 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)

        self.route = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 3),
            nn.GELU(),
            nn.Linear(hidden_dim // 3, self.num_tokens),
            nn.Softmax(dim=-1),
        )
        # 跨模态注意力
        self.cross_attn_mri = nn.MultiheadAttention(mri_dim, num_heads=8)
        self.cross_attn_pet = nn.MultiheadAttention(pet_dim, num_heads=8)

        self.embedding1 = nn.Embedding(hidden_dim, self.num_tokens)  # [64,32] [32, 48] = [64,48]
        self.embedding1.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.embedding3 = nn.Embedding(self.num_tokens, hidden_dim)  # [64,32] [32, 48] = [64,48]
        self.embedding3.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.embedding2 = nn.Embedding(hidden_dim, self.num_tokens)  # [64,32] [32, 48] = [64,48]
        self.embedding2.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.embedding4 = nn.Embedding(self.num_tokens, hidden_dim)  # [64,32] [32, 48] = [64,48]
        self.embedding4.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.align_mri = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2),
                                       nn.GELU(),
                                       nn.Dropout(0.1),
                                       nn.Linear(hidden_dim * 2, hidden_dim)
                                       )
        self.align_pet = nn.Sequential(nn.Linear(pet_dim, hidden_dim * 2),
                                       nn.GELU(),
                                       nn.Dropout(0.1),
                                       nn.Linear(hidden_dim * 2, hidden_dim)
                                       )

        self.sca = Simple_channel_attention(hidden_dim)
        self.ssa = Simple_spatial_attention(hidden_dim)

    def forward(self, mri_feat, pet_feat):
        # 输入特征形状: [B, SeqLen, Dim]

        B, C, H, W = mri_feat.size()

        img1 = self.q1(self.norm1(mri_feat)).permute(0, 2, 3, 1).view(B, H * W, C)
        img2 = self.q2(self.norm2(pet_feat)).permute(0, 2, 3, 1).view(B, H * W, C)

        cat = torch.cat([mri_feat, pet_feat], dim=1)
        cat = self.conv1(cat).permute(0, 2, 3, 1).view(B, H * W, C)
        calss_logits = self.route(cat)
        # calss_logits = F.gumbel_softmax(pred_calss, hard=True, dim=-1)
        # prompt1 = torch.matmul(calss_logits, self.embedding1.weight)
        # prompt2 = torch.matmul(calss_logits, self.embedding2.weight)

        prompt1 = calss_logits * torch.matmul(img2, self.embedding1.weight)
        prompt2 = calss_logits * torch.matmul(img1, self.embedding2.weight)
        prompt1 = torch.matmul(prompt1, self.embedding3.weight)
        prompt2 = torch.matmul(prompt2, self.embedding4.weight)

        mri_attn, _ = self.cross_attn_mri(
            img1, prompt1, prompt1)  # MRI作为Q，PET作为K,V
        pet_attn, _ = self.cross_attn_pet(
            img2, prompt2, prompt2)  # PET作为Q，MRI作为K,V

        # 特征对齐
        aligned_mri = self.align_mri(mri_attn) + mri_attn
        aligned_pet = self.align_pet(pet_attn) + pet_attn

        # 门控融合
        fused = aligned_mri + aligned_pet
        fused = self.sca(fused.permute(0, 2, 1).view(B, C, H, W))
        fused = self.ssa(fused)

        return fused#, prompt2.permute(0, 2, 1).view(B, C, H, W)

class frequency(nn.Module):
    def __init__(self, dim):
        super(frequency, self).__init__()
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')

        self.reduce_1 = nn.Conv2d(dim, dim // 4, kernel_size=1, padding=0)
        self.reduce_2 = nn.Conv2d(dim, dim // 4, kernel_size=1, padding=0)

        self.low_dim = dim // 4
        self.high_dim = dim - (dim // 4)

        self.high_1 = nn.Sequential(nn.Conv2d(self.high_dim, self.high_dim, kernel_size=3, padding=1, groups=self.high_dim),
                                   nn.GELU(),
                                   nn.Conv2d(self.high_dim, self.high_dim, kernel_size=1, padding=0),)
        self.high_2 = nn.Sequential(nn.Conv2d(self.high_dim, self.high_dim, kernel_size=3, padding=1, groups=self.high_dim),
                                   nn.GELU(),
                                   nn.Conv2d(self.high_dim, self.high_dim, kernel_size=1, padding=0),)
        self.low_1 = nn.Sequential(nn.Conv2d(self.low_dim, self.low_dim, kernel_size=3, padding=1, groups=self.low_dim),
                                 nn.GELU(),
                                 nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, padding=0),)
        self.low_2 = nn.Sequential(nn.Conv2d(self.low_dim, self.low_dim, kernel_size=3, padding=1, groups=self.low_dim),
                                 nn.GELU(),
                                 nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, padding=0),)
        
        self.weight_1 = nn.Parameter(torch.ones(self.high_dim, 1, 1), requires_grad=True)
        self.weight_2 = nn.Parameter(torch.ones(self.high_dim, 1, 1), requires_grad=True)

        self.expand = nn.Conv2d(dim // 4, dim, kernel_size=1, padding=0)
        self.sca = Simple_channel_attention(dim)

    def forward(self, mri_feat, pet_feat):
        #cat = self.conv(torch.cat((mri_feat, pet_feat), dim=1))
        feat1 = self.reduce_1(mri_feat)
        feat1_dwt = self.dwt(feat1)
        xl, xh = torch.split(feat1_dwt, [self.low_dim, self.high_dim], dim=1)

        feat2 = self.reduce_2(pet_feat)
        feat2_dwt = self.dwt(feat2)
        yl, yh = torch.split(feat2_dwt, [self.low_dim, self.high_dim], dim=1)

        h1 = self.high_1(xh) * self.weight_1
        l1 = self.low_1(xl)

        h2 = self.high_2(yh) * self.weight_2
        l2 = self.low_2(yl)

        h = h1 + h2
        l = l1 + l2

        idwt = torch.cat([l, h], dim=1)
        out = self.expand(self.idwt(idwt))
        #out = out
        out = self.sca(out)

        return out


class Focal(nn.Module):
    def __init__(self, dim, focal_window=3, focal_level=3, focal_factor=2, bias=True):
        super().__init__()
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor

        #self.act = nn.GELU()
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    #nn.InstanceNorm2d(dim),
                    nn.GELU(),
                )
            )

            self.kernel_sizes.append(kernel_size)

    def forward(self, ctx):
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx

        out = ctx_all

        return out

class SpaExpert(nn.Module):
    def __init__(self, dim, focal_level=3, is_down=False, fusion=True):
        super(SpaExpert, self).__init__()
        self.dim = dim
        self.focal_level = focal_level
        self.is_down = is_down
        self.fusion = fusion
        if self.fusion:
            self.conv1 = nn.Conv2d(dim * 2, dim, kernel_size=1, padding=0)
        else:
            self.conv1 = nn.Identity()

        self.focal = Focal(dim)
        self.act = nn.Sigmoid()

        if self.is_down:
            self.down = nn.AvgPool2d(kernel_size=2, stride=2) # down scale 2

    def forward(self, mri_feat, pet_feat=None):
        C = mri_feat.size(1)

        if self.fusion and pet_feat is not None:
            cat = torch.cat([mri_feat, pet_feat], dim=1)
            cat = self.conv1(cat)
        else:
            cat = mri_feat

        x_v = cat

        if self.is_down:
            x_v = self.down(x_v)
            x_focal = self.focal(x_v)
            x_focal = F.interpolate(x_focal, scale_factor=2, mode='bilinear', align_corners=True)
            x_out   = x_focal * self.act(cat.mean(2, keepdim=True).mean(3, keepdim=True))
        else:
            x_focal = self.focal(x_v)
            x_out   = x_focal * self.act(cat.mean(2, keepdim=True).mean(3, keepdim=True))
        return x_out

class ChannelExpert(nn.Module):
    def __init__(self, dim, focal_level=3, reduction=2, fusion=True):
        super(ChannelExpert, self).__init__()
        self.dim = dim
        self.focal_level = focal_level
        self.fusion = fusion

        self.mid_dim = max(16, dim // reduction)

        if self.fusion:
            self.conv1 = nn.Conv2d(dim * 2, dim, kernel_size=1, padding=0)
        else:
            self.conv1 = nn.Identity()


        self.compress1 =  nn.Conv2d(dim, self.mid_dim, 1, 1, 0)
        self.compress2 =  nn.Conv2d(dim, self.mid_dim, 1, 1, 0)
        self.extend  = nn.Conv2d(self.mid_dim, dim, 1, 1, 0)
        self.focal = Focal(self.mid_dim)
        self.act = nn.Sigmoid()

    def forward(self, mri_feat, pet_feat=None):
        C = mri_feat.size(1)
        if self.fusion and pet_feat is not None:
            cat = torch.cat([mri_feat, pet_feat], dim=1)
            cat = self.conv1(cat)
        else:
            cat = mri_feat

        x_v = cat
        x_v = self.compress2(x_v)
        x_focal = self.focal(x_v)
        x_out = self.extend(x_focal) * self.act(cat.mean(2, keepdim=True).mean(3, keepdim=True))
        return x_out

# class Expert(nn.Module):
#     """差异化卷积专家网络"""
#
#     def __init__(self, expert_type, in_channels, out_channels, fusion=True):
#         super().__init__()
#         self.type = expert_type
#
#         if expert_type == "modal_sense":
#             self.expert = ASGFM(in_channels, in_channels, in_channels)
#         elif expert_type == "frequency":
#             self.expert = frequency(in_channels)
#         elif expert_type == "spatial":
#             if fusion:
#                 self.expert = SpaExpert(in_channels, is_down=False, fusion=True)
#             else:
#                 self.expert = SpaExpert(in_channels, is_down=False, fusion=False)
#         elif expert_type == "spatial_down":
#             if fusion:
#                 self.expert = SpaExpert(in_channels, is_down=True, fusion=True)
#             else:
#                 self.expert = SpaExpert(in_channels, is_down=True, fusion=False)
#         elif expert_type == "channel_r2":
#             if fusion:
#                 self.expert = ChannelExpert(in_channels, reduction=2, fusion=True)
#             else:
#                 self.expert = ChannelExpert(in_channels, reduction=2, fusion=False)
#         elif expert_type == "channel_r4":
#             if fusion:
#                 self.expert = ChannelExpert(in_channels, reduction=4, fusion=True)
#             else:
#                 self.expert = ChannelExpert(in_channels, reduction=4, fusion=False)
#         elif expert_type == "channel_r8":
#             if fusion:
#                 self.expert = ChannelExpert(in_channels, reduction=8, fusion=True)
#             else:
#                 self.expert = ChannelExpert(in_channels, reduction=8, fusion=False)
#         elif expert_type == "channel_r16":
#             if fusion:
#                 self.expert = ChannelExpert(in_channels, reduction=16, fusion=True)
#             else:
#                 self.expert = ChannelExpert(in_channels, reduction=16, fusion=False)
#         else:
#             raise ValueError("Unsupported expert type")
#
#     def forward(self, mri_feat, pet_feat):
#         if self.type == "modal_sense":
#             return self.expert(mri_feat, pet_feat)
#         elif self.type == "frequency":
#             return self.expert(mri_feat, pet_feat)
#         elif self.type == "spatial":
#             return self.expert(mri_feat, pet_feat)
#         elif self.type == "spatial_down":
#             return self.expert(mri_feat, pet_feat)
#         elif self.type == "channel_r2":
#             return self.expert(mri_feat, pet_feat)
#         elif self.type == "channel_r4":
#             return self.expert(mri_feat, pet_feat)
#         elif self.type == "channel_r8":
#             return self.expert(mri_feat, pet_feat)
#         elif self.type == "channel_r16":
#             return self.expert(mri_feat, pet_feat)

class SMoEFusion(nn.Module):
    """图像特征MoE融合模块"""

    def __init__(self,
                 in_channels,  # 特征1的通道数
                 num_experts=6,
                 expert_out_channels=64,
                 top_k=2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.total_channels = in_channels

        # 创建差异化专家
        # expert_types = ["modal_sense", "frequency", "spatial", "spatial_down", "channel_r4", "channel_r8"]
        # self.experts = nn.ModuleList([
        #     Expert(expert_types[i], self.total_channels, self.total_channels)
        #     for i in range(num_experts)
        # ])

        self.agg_experts = nn.ModuleList([
            ASGFM(self.total_channels, self.total_channels, self.total_channels),
            frequency(self.total_channels)])
        
        self.experts1 = nn.ModuleList([
            SpaExpert(self.total_channels, is_down=False, fusion=False),
            ChannelExpert(self.total_channels, reduction=4, fusion=False),])
        
        self.experts2 = nn.ModuleList([
            SpaExpert(self.total_channels, is_down=False, fusion=False),
            ChannelExpert(self.total_channels, reduction=4, fusion=False),])
        
        # 空间门控网络
        self.gate_gen = nn.Sequential(
            nn.Conv2d(self.total_channels * 2, self.total_channels, 1, padding=0),
            nn.Conv2d(self.total_channels, self.total_channels, 3, padding=1, groups=self.total_channels),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(self.total_channels)
        )
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Sequential(nn.Linear(self.total_channels, self.total_channels // 2),
                                 #nn.Dropout(0.1),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(self.total_channels // 2, num_experts),
                                 nn.Softmax(dim=1))

        self.proj = nn.Conv2d(self.total_channels, self.total_channels, kernel_size=3, padding=1)

        #self.sp = nn.Softplus()
        self._init_weights()

    def _init_weights(self):
        """初始化MoE模块的权重"""
        # 初始化门控网络的卷积层
        for m in self.gate_gen.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 初始化门控网络的FC层
        # 使用较小的标准差初始化第一层
        nn.init.normal_(self.fc1[0].weight, std=0.01)
        nn.init.constant_(self.fc1[0].bias, 0)
        
        # 使用零初始化最后一层，促进训练初期平等使用专家
        nn.init.zeros_(self.fc1[2].weight)
        #nn.init.constant_(self.fc1[2].bias, 0)
        
        # 初始化投影层
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out')
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, feat1, feat2):
        """
        输入:
            feat1: [B, C1, H, W]
            feat2: [B, C2, H, W]
        输出:
            fused_feature: [B, Cout, H, W]
        """
        # 通道维度拼接特征

        # 生成门控权重 [B, num_experts]
        combined = torch.cat([feat1, feat2], dim=1)
        gate = self.gate_gen(combined)
        B, C, _, _ = gate.shape
        gate = self.pool_avg(gate) + self.pool_max(gate)
        gate = gate.view(B, C)
        gate_weights = self.fc1(gate)

        #print(gate_weights)

        # 计算各专家输出
        expert_outputs = []
        out = torch.zeros_like(feat1).to(feat1.device)
        for idx, expert in enumerate(self.agg_experts):

            expert_out = expert(feat1, feat2)  # 每个输出都是[B, Cout, H, W]
            cof_k = gate_weights[:, idx].view(-1, 1, 1, 1)
            out +=  cof_k * expert_out
            expert_outputs.append(expert_out.unsqueeze(1))  # [B, 1, Cout, H, W]

        for idx, expert in enumerate(self.experts1):

            expert_out_1 = expert(feat1, None)
            cof_k = gate_weights[:, idx + 2].view(-1, 1, 1, 1)
            out +=  cof_k * expert_out_1
            expert_outputs.append(expert_out_1.unsqueeze(1))  # [B, 1, Cout, H, W]
        
        for idx, expert in enumerate(self.experts2):

            expert_out_2 = expert(feat2, None)
            cof_k = gate_weights[:, idx + 4].view(-1, 1, 1, 1)
            out +=  cof_k * expert_out_2
            expert_outputs.append(expert_out_2.unsqueeze(1))


        # 堆叠专家输出 [B, num_experts, Cout, H, W]
        #expert_stack = torch.cat(expert_outputs, dim=1)
        #print(expert_stack.shape)
        # # 加权融合
        # weights = gate_weights.view(-1, self.num_experts, 1, 1, 1)  # 扩展维度
        # fused = (expert_stack * weights).sum(dim=1)  # [B, Cout, H, W]
        out = self.proj(out)

        return out

##########################################################################
##---------- Restormer -----------------------
class MRCENet(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=48,
                 num_blocks=[2, 4, 4, 8],
                 num_refinement_blocks=2,
                 heads=[4, 4, 8, 8], # n2
                 windows_size=8,
                 #ffn_expansion_factor=[2.66, 2.66, 1, 1], # n2
                 ffn_expansion_factor=[2.66, 2.66, 2.66, 2.66], # n5_f2
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):
        super(MRCENet, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)


        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor[0], bias=bias,
                             LayerNorm_type=LayerNorm_type, shift_size=0 if (i % 2 ==0) else windows_size, window_size=windows_size) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim, int(dim * 2 ** 1))  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor[1], bias=bias,
                             LayerNorm_type=LayerNorm_type, shift_size=0 if (i % 2 ==0) else windows_size, window_size=windows_size) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1), int(dim * 2 ** 2))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor[2],
                  bias=bias, LayerNorm_type=LayerNorm_type, shift_size=0 if (i % 2 ==0) else windows_size, window_size=windows_size) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2), int(dim * 2 ** 3))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor[3],
                  bias=bias, LayerNorm_type=LayerNorm_type, shift_size=0 if (i % 2 ==0) else windows_size, window_size=windows_size) for i in range(num_blocks[3])])
        
        self.patch_embed_2 = OverlapPatchEmbed(inp_channels, dim)
        # self.patch_embed_2 = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1_ = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor[0], bias=bias,
                             LayerNorm_type=LayerNorm_type, shift_size=0 if (i % 2 ==0) else windows_size, window_size=windows_size) for i in range(num_blocks[0])])

        self.down1_2_ = Downsample(dim, int(dim * 2 ** 1))  ## From Level 1 to Level 2
        self.encoder_level2_ = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor[1], bias=bias,
                             LayerNorm_type=LayerNorm_type, shift_size=0 if (i % 2 ==0) else windows_size, window_size=windows_size) for i in range(num_blocks[1])])

        self.down2_3_ = Downsample(int(dim * 2 ** 1), int(dim * 2 ** 2))  ## From Level 2 to Level 3
        self.encoder_level3_ = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor[2],
                  bias=bias, LayerNorm_type=LayerNorm_type, shift_size=0 if (i % 2 ==0) else windows_size, window_size=windows_size) for i in range(num_blocks[2])])

        self.down3_4_ = Downsample(int(dim * 2 ** 2), int(dim * 2 ** 3))  ## From Level 3 to Level 4
        self.latent_ = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor[3],
                  bias=bias, LayerNorm_type=LayerNorm_type, shift_size=0 if (i % 2 ==0) else windows_size, window_size=windows_size) for i in range(num_blocks[3])])


        self.MRCE1 = SMoEFusion(in_channels=int(dim * 2 ** 3), num_experts=6, top_k=2)
        self.MRCE2 = SMoEFusion(in_channels=int(dim * 2 ** 2), num_experts=6, top_k=2)
        self.MRCE3 = SMoEFusion(in_channels=int(dim * 2 ** 1), num_experts=6, top_k=2)
        self.MRCE4 = SMoEFusion(in_channels=dim, num_experts=6, top_k=2)

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor[2],
                  bias=bias, LayerNorm_type=LayerNorm_type, shift_size=0 if (i % 2 ==0) else windows_size,  window_size=windows_size) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor[1],
                  bias=bias, LayerNorm_type=LayerNorm_type, shift_size=0 if (i % 2 ==0) else windows_size, window_size=windows_size) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor[0],
                  bias=bias, LayerNorm_type=LayerNorm_type, shift_size=0 if (i % 2 ==0) else windows_size, window_size=windows_size) for i in range(num_blocks[0])])
        
        #self.up1_1 = Upsample(int(dim * 2 ** 1))
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor[0],
                  bias=bias, LayerNorm_type=LayerNorm_type, shift_size=0, window_size=windows_size) for i in range(num_refinement_blocks)])


        self.output = nn.Sequential(  #nn.ReflectionPad2d(3),
            # nn.GroupNorm(num_groups=32, num_channels=dim * 2 ** 1),
            nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def encoder_1(self, input):
        inp_enc_level1 = self.patch_embed(input)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        return latent, out_enc_level3, out_enc_level2, out_enc_level1, inp_enc_level1

    def encoder_2(self, input):
        inp_enc_level1 = self.patch_embed_2(input)
        out_enc_level1 = self.encoder_level1_(inp_enc_level1)

        inp_enc_level2 = self.down1_2_(out_enc_level1)
        out_enc_level2 = self.encoder_level2_(inp_enc_level2)

        inp_enc_level3 = self.down2_3_(out_enc_level2)
        out_enc_level3 = self.encoder_level3_(inp_enc_level3)

        inp_enc_level4 = self.down3_4_(out_enc_level3)
        latent = self.latent_(inp_enc_level4)
        return latent, out_enc_level3, out_enc_level2, out_enc_level1, inp_enc_level1

    def decder(self, latent, out_enc_level3, out_enc_level2, out_enc_level1):
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        return out_dec_level1

    def forward(self, inp_img, img_2):
        latent1, out_enc_level31, out_enc_level21, out_enc_level11, inp_enc_level11 = self.encoder_1(inp_img)
        latent2, out_enc_level32, out_enc_level22, out_enc_level12, inp_enc_level12 = self.encoder_2(img_2)


        latent = self.MRCE1(latent1, latent2)

        out_dec_level3 = self.MRCE2(out_enc_level31, out_enc_level32)

        out_dec_level2 =  self.MRCE3(out_enc_level21, out_enc_level22)

        out_dec_level1 = self.MRCE4(out_enc_level11, out_enc_level12)

        out_dec_level1 = self.decder(latent, out_dec_level3, out_dec_level2, out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1


if __name__ == "__main__":
    # from fvcore.nn import FlopCountAnalysis, flop_count_table
    img1 = torch.randn(1, 1, 128, 128)
    img2 = torch.randn(1, 1, 128, 128)
    model = MRCENet()
    y = model(img1, img2)
    # flops = FlopCountAnalysis(model, img)
    print('output shape:', y.shape)
    # print('flops = ', flops.total() / 1e9)
    # print(flop_count_table(flops))
    from thop import profile

    flops, params = profile(model, inputs=(img1,img2))
    print('Params and FLOPs are {}M/{}G'.format(params / 1e6, flops / 1e9))

    i = torch.randn(1, 64, 64, 64)
    down = nn.PixelUnshuffle(2)
    out = down(i)
    print(out.shape)