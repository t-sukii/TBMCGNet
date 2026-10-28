import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


# from networks.Gt_u_net.GT_UNet import MHSA


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1, groups=1, has_bn=True,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize, stride=stride, padding=pad, dilation=dilation,
                              groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(hidden_features)
        self.norm3 = nn.BatchNorm2d(dim)
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.project_in(x)
        x = self.gelu1(x)
        x = self.norm1(x)
        x = self.dwconv(x)
        x = self.gelu2(x)
        x = self.norm2(x)
        x = self.project_out(x)
        x = self.norm3(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.sr1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=2, stride=2, groups=dim, bias=True),
        )
        self.sr2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=2, stride=2, groups=dim, bias=True),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = self.sr1(q)
        k = self.sr2(k)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768, out_dim=1):
        super(DWConv, self).__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, out_dim, 1, padding=0, bias=False),
        )

    def forward(self, x):
        x = self.dwconv(x)
        return x


class UnetBasicBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, stride=1, padding=padding)
        self.norm2 = nn.BatchNorm2d(out_dim)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        return out


class DualResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(out_dim)
        self.conv5 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x = x1 + x2
        x = self.relu1(x)
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x = x + x3
        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x5 = self.conv5(x)
        x5 = self.bn5(x5)
        out = x4 + x5
        out = self.relu2(out)
        return out


class DualDWResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.conv3 = DWConv(out_dim, out_dim)
        self.bn3 = nn.BatchNorm2d(out_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(out_dim)
        self.conv5 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x = x1 + x2
        x = self.relu1(x)
        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x = x + x3
        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x5 = self.conv5(x)
        x5 = self.bn5(x5)
        out = x4 + x5
        out = self.relu2(out)
        return out


class UnetUpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(in_dim, out_dim, 2, 2)
        self.conv = nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        out = self.trans_conv(out)
        out = self.conv(out)
        out = self.relu(out)
        out = self.norm(out)
        return out


class OutBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 1, 1)

    def forward(self, x):
        out = self.conv(x)
        return out
