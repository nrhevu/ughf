import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch._jit_internal import Optional
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class DOConv2d(Module):
    """
    DOConv2d can be used as an alternative for torch.nn.Conv2d.
    The interface is similar to that of Conv2d, with one exception:
         1. D_mul: the depth multiplier for the over-parameterization.
    Note that the groups parameter switchs between DO-Conv (groups=1),
    DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
        "D_mul",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        D_mul=None,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
        simam=False,
    ):
        super(DOConv2d, self).__init__()

        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        padding = (padding, padding)
        dilation = (dilation, dilation)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(
                    valid_padding_modes, padding_mode
                )
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        self.simam = simam
        #################################### Initailization of D & W ###################################
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.W = Parameter(
            torch.Tensor(out_channels, in_channels // groups, self.D_mul)
        )
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(
                torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N)
            )
            D_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.D_diag = Parameter(
                    torch.cat([D_diag, zeros], dim=2), requires_grad=False
                )
            else:  # the case when D_mul = M * N
                self.D_diag = Parameter(D_diag, requires_grad=False)
        ##################################################################################################
        if simam:
            self.simam_block = simam_module()
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"

    def _conv_forward(self, input, weight):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                weight,
                self.bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if M * N > 1:
            ######################### Compute DoW #################
            # (input_channels, D_mul, M * N)
            D = self.D + self.D_diag
            W = torch.reshape(
                self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul)
            )

            # einsum outputs (out_channels // groups, in_channels, M * N),
            # which is reshaped to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(torch.einsum("ims,ois->oim", D, W), DoW_shape)
            #######################################################
        else:
            DoW = torch.reshape(self.W, DoW_shape)
        if self.simam:
            DoW_h1, DoW_h2 = torch.chunk(DoW, 2, dim=2)
            DoW = torch.cat([self.simam_block(DoW_h1), DoW_h2], dim=2)

        return self._conv_forward(input, DoW)


class DOConv2d_eval(Module):
    """
    DOConv2d can be used as an alternative for torch.nn.Conv2d.
    The interface is similar to that of Conv2d, with one exception:
         1. D_mul: the depth multiplier for the over-parameterization.
    Note that the groups parameter switchs between DO-Conv (groups=1),
    DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
        "D_mul",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        D_mul=None,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
        simam=False,
    ):
        super(DOConv2d_eval, self).__init__()

        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        padding = (padding, padding)
        dilation = (dilation, dilation)

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(
                    valid_padding_modes, padding_mode
                )
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        self.simam = simam
        #################################### Initailization of D & W ###################################
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, M, N))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        self.register_parameter("bias", None)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"

    def _conv_forward(self, input, weight):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                weight,
                self.bias,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, input):
        return self._conv_forward(input, self.W)


class simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )
        return x * self.activaton(y)


def _zigzag_low_freq_coords(h: int, w: int, k: int) -> Tuple[List[int], List[int]]:
    """
    Pick k (u,v) frequency coordinates from the low-frequency corner (0,0)
    using a simple zig-zag scan over diagonals (JPEG-like).
    """
    coords = []
    for s in range(h + w - 1):
        # diagonal s: (u, v) with u+v=s
        diag = []
        u_min = max(0, s - (w - 1))
        u_max = min(h - 1, s)
        for u in range(u_min, u_max + 1):
            v = s - u
            diag.append((u, v))
        # alternate direction for zigzag
        if s % 2 == 1:
            diag.reverse()
        coords.extend(diag)
        if len(coords) >= k:
            break
    coords = coords[:k]
    mapper_x = [u for (u, v) in coords]
    mapper_y = [v for (u, v) in coords]
    return mapper_x, mapper_y


def _build_dct_filter(tile: int, freq: int, pos: int) -> float:
    """
    1D DCT-II basis value at position pos for frequency freq and length tile.
    """
    val = math.cos(math.pi * freq * (pos + 0.5) / tile) / math.sqrt(tile)
    return val if freq == 0 else val * math.sqrt(2.0)


def _make_dct_weights(
    dct_h: int,
    dct_w: int,
    mapper_x: List[int],
    mapper_y: List[int],
    channels: int,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """
    Create fixed DCT weights of shape [C, dct_h, dct_w].
    Channels are split evenly across selected frequency components.
    """
    num_freq = len(mapper_x)
    assert num_freq == len(mapper_y)
    assert (
        channels % num_freq == 0
    ), f"channels ({channels}) must be divisible by num_freq ({num_freq})"
    c_part = channels // num_freq

    weight = torch.zeros((channels, dct_h, dct_w), device=device, dtype=dtype)
    for i, (u, v) in enumerate(zip(mapper_x, mapper_y)):
        for x in range(dct_h):
            bx = _build_dct_filter(dct_h, u, x)
            for y in range(dct_w):
                by = _build_dct_filter(dct_w, v, y)
                weight[i * c_part : (i + 1) * c_part, x, y] = bx * by
    return weight


class MultiSpectralAttentionLayer(nn.Module):
    """
    Multi-Spectral (DCT) Channel Attention (FcaNet-style).

    Forward:
      x: [B, C, H, W]
      1) adaptive pool -> [B, C, dct_h, dct_w]
      2) DCT projection (fixed weights) -> y: [B, C]
      3) FC -> sigmoid gates -> [B, C, 1, 1]
      4) x * gates
    """

    def __init__(
        self,
        channels: int,
        dct_h: int = 7,
        dct_w: int = 7,
        num_freq: int = 16,
        reduction: int = 16,
        freq_coords: Tuple[List[int], List[int]] | None = None,
    ):
        super().__init__()
        assert num_freq in (1, 2, 4, 8, 16, 32), "common choices: 1,2,4,8,16,32"
        self.channels = channels
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.num_freq = num_freq

        if freq_coords is None:
            mapper_x, mapper_y = _zigzag_low_freq_coords(dct_h, dct_w, num_freq)
        else:
            mapper_x, mapper_y = freq_coords
            assert len(mapper_x) == num_freq and len(mapper_y) == num_freq

        # fixed DCT weights as a buffer
        dct_weight = _make_dct_weights(dct_h, dct_w, mapper_x, mapper_y, channels)
        self.register_buffer("dct_weight", dct_weight)

        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert c == self.channels, f"expected C={self.channels}, got {c}"

        # pool to the DCT tile size (handles arbitrary H,W)
        x_pooled = F.adaptive_avg_pool2d(
            x, (self.dct_h, self.dct_w)
        )  # [B,C,dct_h,dct_w]

        # DCT projection: sum_{x,y} (feat * dct_weight)
        # -> [B,C]
        y = (x_pooled * self.dct_weight).sum(dim=(2, 3))

        # channel gates
        g = self.fc(y).view(b, c, 1, 1)
        return x * g


def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = (
        x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    )
    return windows


def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print('B: ', B)
    # print(H // window_size)
    # print(W // window_size)
    C = windows.shape[1]
    # print('C: ', C)
    x = windows.view(
        -1, H // window_size, W // window_size, C, window_size, window_size
    )
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x


def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]
        b_dd = x_dd.shape[0] + b_d
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]


def window_reversex(windows, window_size, H, W, batch_list):
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_reverses(windows[: batch_list[0], ...], window_size, h, w)
    B, C, _, _ = x_main.shape
    # print('windows: ', windows.shape)
    # print('batch_list: ', batch_list)
    res = torch.zeros([B, C, H, W], device=windows.device)
    res[:, :, :h, :w] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverses(
            windows[batch_list[2] :, ...], window_size, window_size, window_size
        )
        res[:, :, h:, w:] = x_dd[:, :, h - H :, w - W :]
        x_r = window_reverses(
            windows[batch_list[0] : batch_list[1], ...], window_size, h, window_size
        )
        res[:, :, :h, w:] = x_r[:, :, :, w - W :]
        x_d = window_reverses(
            windows[batch_list[1] : batch_list[2], ...], window_size, window_size, w
        )
        res[:, :, h:, :w] = x_d[:, :, h - H :, :]
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverses(
            windows[batch_list[0] : batch_list[1], ...], window_size, h, window_size
        )
        res[:, :, :h, w:] = x_r[:, :, :, w - W :]
    if h != H and len(batch_list) == 2:
        x_d = window_reverses(
            windows[batch_list[0] : batch_list[1], ...], window_size, window_size, w
        )
        res[:, :, h:, :w] = x_d[:, :, h - H :, :]
    return res


class BasicConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        bias=False,
        norm=False,
        relu=True,
        transpose=False,
        channel_shuffle_g=0,
        norm_method=nn.BatchNorm2d,
        groups=1,
    ):
        super(BasicConv, self).__init__()
        self.channel_shuffle_g = channel_shuffle_g
        self.norm = norm
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                    groups=groups,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                    groups=groups,
                )
            )
        if norm:
            layers.append(norm_method(out_channel))
        elif relu:
            layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BasicConv_do(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        bias=False,
        norm=False,
        relu=True,
        transpose=False,
        relu_method=nn.ReLU,
        groups=1,
        norm_method=nn.BatchNorm2d,
    ):
        super(BasicConv_do, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                )
            )
        else:
            layers.append(
                DOConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                    groups=groups,
                )
            )
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(
                out_channel, out_channel, kernel_size=3, stride=1, relu=True, norm=False
            ),
            BasicConv(
                out_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                relu=False,
                norm=False,
            ),
        )

    def forward(self, x):
        return self.main(x) + x


class ResBlock_do(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock_do, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
        )

    def forward(self, x):
        return self.main(x) + x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class ResBlock_do_fft_bench(nn.Module):
    def __init__(self, out_channel, norm="backward"):
        super(ResBlock_do_fft_bench, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False),
        )
        self.main_fft = nn.Sequential(
            BasicConv_do(
                out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True
            ),
            BasicConv_do(
                out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False
            ),
        )
        self.dim = out_channel
        self.norm = norm
        self.LayerNorm = nn.LayerNorm(out_channel, elementwise_affine=True)
        self.conv1 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel * 2,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        x1 = self.LayerNorm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = torch.fft.rfft2(x1, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y1 = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = self.LayerNorm(y1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = self.conv1(y)
        y = self.sg(y)
        y = self.conv2(y) + y1
        x1 = self.main(x)

        return x1 + x + y
