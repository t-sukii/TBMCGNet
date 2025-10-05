import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from thop import profile
from ptflops import get_model_complexity_info
import time

from _blocks import UnetBasicBlock, DualResBlock, ConvBlock, DWConv, DualDWResBlock

BATCH_NORM_DECAY = 1 - 0.9  # pytorch batch norm `momentum = 1 - counterpart` of tensorflow
BATCH_NORM_EPSILON = 1e-5


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, dim: int = -2) -> torch.Tensor:
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)

    return y_soft


class PAM(nn.Module):
    def __init__(self):
        super(PAM, self).__init__()

    def forward(self, a, b, c):
        mid = F.softmax(b @ c.transpose(-2, -1), dim=-1)
        out = a @ mid.transpose(-2, -1)
        out += a
        return out


class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, a, b, c):
        mid = F.softmax(b.view(b.size(0), b.size(1), -1) @ c.view(c.size(0), c.size(1), -1).transpose(1, 2), dim=-1)
        out = mid @ a.view(a.size(0), a.size(1), -1)
        out = out.view(a.size())
        out += a
        return out


def get_act(activation):
    """Only supports ReLU and SiLU/Swish."""
    assert activation in ['relu', 'silu']
    if activation == 'relu':
        return nn.ReLU()
    else:
        return nn.Hardswish()  # TODO: pytorch's nn.Hardswish() v.s. tf.nn.swish


class BNReLU(nn.Module):
    """"""

    def __init__(self, out_channels, activation='relu', nonlinearity=True, init_zero=False):
        super(BNReLU, self).__init__()

        self.norm = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_DECAY, eps=BATCH_NORM_EPSILON)
        if nonlinearity:
            self.act = get_act(activation)
        else:
            self.act = None

        if init_zero:
            nn.init.constant_(self.norm.weight, 0)
        else:
            nn.init.constant_(self.norm.weight, 1)

    def forward(self, input):
        out = self.norm(input)
        if self.act is not None:
            out = self.act(out)
        return out


class RelPosSelfAttention(nn.Module):
    """Relative Position Self Attention"""

    def __init__(self, h, w, dim, relative=True, fold_heads=False):
        super(RelPosSelfAttention, self).__init__()
        self.relative = relative
        self.fold_heads = fold_heads
        self.rel_emb_w = nn.Parameter(torch.Tensor(2 * w - 1, dim))
        self.rel_emb_h = nn.Parameter(torch.Tensor(2 * h - 1, dim))

        nn.init.normal_(self.rel_emb_w, std=dim ** -0.5)
        nn.init.normal_(self.rel_emb_h, std=dim ** -0.5)

    def forward(self, q, k, v):
        """2D self-attention with rel-pos. Add option to fold heads."""
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)  # scaled dot-product
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        if self.relative:
            logits += self.relative_logits(q)
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = F.softmax(weights, dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def relative_logits(self, q):
        # Relative logits in width dimension.
        rel_logits_w = self.relative_logits_1d(q, self.rel_emb_w, transpose_mask=[0, 1, 2, 4, 3, 5])
        # Relative logits in height dimension
        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.rel_emb_h,
                                               transpose_mask=[0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w

    def relative_logits_1d(self, q, rel_k, transpose_mask):
        bs, heads, h, w, dim = q.shape
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, [-1, heads * h, w, 2 * w - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = torch.reshape(rel_logits, [-1, heads, h, w, w])
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat(1, 1, 1, h, 1, 1)
        rel_logits = rel_logits.permute(*transpose_mask)
        return rel_logits

    def rel_to_abs(self, x):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, length, 2*length - 1]
        Output: [bs, heads, length, length]
        """
        bs, heads, length, _ = x.shape
        col_pad = torch.zeros((bs, heads, length, 1), dtype=x.dtype).cuda()
        x = torch.cat([x, col_pad], dim=3)
        flat_x = torch.reshape(x, [bs, heads, -1]).cuda()
        flat_pad = torch.zeros((bs, heads, length - 1), dtype=x.dtype).cuda()
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)
        final_x = torch.reshape(
            flat_x_padded, [bs, heads, length + 1, 2 * length - 1])
        final_x = final_x[:, :, :length, length - 1:]
        return final_x


class AbsPosSelfAttention(nn.Module):

    def __init__(self, W, H, dkh, absolute=True, fold_heads=False):
        super(AbsPosSelfAttention, self).__init__()
        self.absolute = absolute
        self.fold_heads = fold_heads

        self.emb_w = nn.Parameter(torch.Tensor(W, dkh))
        self.emb_h = nn.Parameter(torch.Tensor(H, dkh))
        nn.init.normal_(self.emb_w, dkh ** -0.5)
        nn.init.normal_(self.emb_h, dkh ** -0.5)

    def forward(self, q, k, v):
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)  # scaled dot-product
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        abs_logits = self.absolute_logits(q)
        if self.absolute:
            logits += abs_logits
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = F.softmax(weights, dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def absolute_logits(self, q):
        """Compute absolute position enc logits."""
        emb_h = self.emb_h[:, None, :]
        emb_w = self.emb_w[None, :, :]
        emb = emb_h + emb_w
        abs_logits = torch.einsum('bhxyd,pqd->bhxypq', q, emb)
        return abs_logits


class GroupPointWise(nn.Module):
    """"""

    def __init__(self, in_channels, heads=4, proj_factor=1, target_dimension=None):
        super(GroupPointWise, self).__init__()
        if target_dimension is not None:
            proj_channels = target_dimension // proj_factor
        else:
            proj_channels = in_channels // proj_factor
        self.w = nn.Parameter(
            torch.Tensor(in_channels, heads, proj_channels // heads)
        )

        nn.init.normal_(self.w, std=0.01)

    def forward(self, input):
        # dim order:  pytorch BCHW v.s. TensorFlow BHWC
        input = input.permute(0, 2, 3, 1).float()
        """
        b: batch size
        h, w : imput height, width
        c: input channels
        n: num head
        p: proj_channel // heads
        """
        out = torch.einsum('bhwc,cnp->bnhwp', input, self.w)
        return out


class MHSA(nn.Module):

    def __init__(self, in_channels, heads, curr_h, curr_w, pos_enc_type='relative', use_pos=True):
        super(MHSA, self).__init__()
        self.q_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.k_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.v_proj = GroupPointWise(in_channels, heads, proj_factor=1)

        assert pos_enc_type in ['relative', 'absolute']
        if pos_enc_type == 'relative':
            self.self_attention = RelPosSelfAttention(curr_h, curr_w, in_channels // heads, fold_heads=True)
        else:
            raise NotImplementedError

    def forward(self, input):
        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)

        o = self.self_attention(q=q, k=k, v=v)
        return o


class BotBlock(nn.Module):

    def __init__(self, in_dimension, curr_h, curr_w, proj_factor=4, activation='relu', pos_enc_type='relative',
                 stride=1, target_dimension=None):
        super(BotBlock, self).__init__()

        bottleneck_dimension = target_dimension // proj_factor
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dimension, bottleneck_dimension, kernel_size=3, padding=1, stride=1),
            BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True)
        )

        self.mhsa = MHSA(in_channels=bottleneck_dimension, heads=4, curr_h=curr_h, curr_w=curr_w,
                         pos_enc_type=pos_enc_type)
        conv2_list = []
        # if stride != 1:
        #     assert stride == 2, stride
        #     conv2_list.append(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)))  # TODO: 'same' in tf.pooling
        conv2_list.append(BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True))
        self.conv2 = nn.Sequential(*conv2_list)

        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_dimension, target_dimension, kernel_size=3, padding=1, stride=1),
            BNReLU(target_dimension, nonlinearity=False, init_zero=True),
        )
        self.last_act = get_act(activation)

    def forward(self, x):
        # Q_h = Q_w = 4
        # N, C, H, W = x.shape
        # P_h, P_w = H // Q_h, W // Q_w
        #
        # x = x.reshape(N * P_h * P_w, C, Q_h, Q_w)
        #
        # out = self.conv1(x)
        # out = self.mhsa(out)
        # out = out.permute(0, 3, 1, 2)  # back to pytorch dim order
        # out = self.conv2(out)
        # out = self.conv3(out)
        #
        # N1, C1, H1, W1 = out.shape
        # out = out.reshape(N, C1, int(H), int(W))
        original_size = x.shape[-2:]

        Q_h = Q_w = 4
        _, _, H, W = x.shape
        pad_input = (H % Q_h != 0) or (W % Q_w != 0)
        if pad_input:
            x = F.pad(x, (0, Q_w - W % Q_w, 0, Q_h - H % Q_h, 0, 0))
        N, C, H, W = x.shape
        P_h, P_w = H // Q_h, W // Q_w

        x = x.reshape(N * P_h * P_w, C, Q_h, Q_w)

        out = self.conv1(x)
        out = self.mhsa(out)
        out = out.permute(0, 3, 1, 2)  # back to pytorch dim order
        out = self.conv2(out)
        out = self.conv3(out)

        N1, C1, H1, W1 = out.shape
        out = out.reshape(N, C1, int(H), int(W))
        if pad_input:
            out = out[:, :, :original_size[0], :original_size[1]]

        # out += shortcut
        out = self.last_act(out)

        return out


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


def _make_bot_layer(ch_in, ch_out):
    W = H = 4
    dim_in = ch_in
    dim_out = ch_out

    stage5 = []

    stage5.append(
        BotBlock(in_dimension=dim_in, curr_h=H, curr_w=W, stride=1, target_dimension=dim_out)
    )

    return nn.Sequential(*stage5)


class conv_out_layer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        out = self.relu(x)
        return out


class net_7(nn.Module):  # This code is MS-res conv + trans, with Channel attention
    def __init__(self, img_ch=3, output_ch=1, embed_dim=32):
        super(net_7, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.trans1 = _make_bot_layer(ch_in=img_ch, ch_out=embed_dim * 2)
        # self.Conv1 = UnetBasicBlock(in_dim=img_ch, out_dim=embed_dim * 2, kernel_size=3, stride=1, padding=1)
        self.Conv1 = DualResBlock(in_dim=img_ch, out_dim=embed_dim * 2)
        # self.fuse1 = ConvBlock(in_channels=embed_dim * 4, out_channels=embed_dim * 2)
        self.fuse1 = DWConv(dim=embed_dim * 4, out_dim=embed_dim * 2)

        self.trans2 = _make_bot_layer(ch_in=embed_dim * 2, ch_out=embed_dim * 4)
        # self.Conv2 = UnetBasicBlock(in_dim=embed_dim * 2, out_dim=embed_dim * 4, kernel_size=3, stride=1, padding=1)
        self.Conv2 = DualResBlock(in_dim=embed_dim * 2, out_dim=embed_dim * 4)
        # self.fuse2 = ConvBlock(in_channels=embed_dim * 8, out_channels=embed_dim * 4)
        self.fuse2 = DWConv(dim=embed_dim * 8, out_dim=embed_dim * 4)

        self.trans3 = _make_bot_layer(ch_in=embed_dim * 4, ch_out=embed_dim * 8)
        # self.Conv3 = UnetBasicBlock(in_dim=embed_dim * 4, out_dim=embed_dim * 8, kernel_size=3, stride=1, padding=1)
        self.Conv3 = DualResBlock(in_dim=embed_dim * 4, out_dim=embed_dim * 8)
        # self.fuse3 = ConvBlock(in_channels=embed_dim * 16, out_channels=embed_dim * 8)
        self.fuse3 = DWConv(dim=embed_dim * 16, out_dim=embed_dim * 8)

        self.trans4 = _make_bot_layer(ch_in=embed_dim * 8, ch_out=embed_dim * 16)
        # self.Conv4 = UnetBasicBlock(in_dim=embed_dim * 8, out_dim=embed_dim * 16, kernel_size=3, stride=1, padding=1)
        self.Conv4 = DualResBlock(in_dim=embed_dim * 8, out_dim=embed_dim * 16)
        # self.fuse4 = ConvBlock(in_channels=embed_dim * 32, out_channels=embed_dim * 16)
        self.fuse4 = DWConv(dim=embed_dim * 32, out_dim=embed_dim * 16)

        self.trans5 = _make_bot_layer(ch_in=embed_dim * 16, ch_out=embed_dim * 32)
        # self.Conv5 = UnetBasicBlock(in_dim=embed_dim * 16, out_dim=embed_dim * 32, kernel_size=3, stride=1, padding=1)
        self.Conv5 = DualResBlock(in_dim=embed_dim * 16, out_dim=embed_dim * 32)
        # self.fuse5 = ConvBlock(in_channels=embed_dim * 64, out_channels=embed_dim * 32)
        self.fuse5 = DWConv(dim=embed_dim * 64, out_dim=embed_dim * 32)

        self.Up5 = up_conv(ch_in=embed_dim * 32, ch_out=embed_dim * 16)
        self.Up_trans5 = _make_bot_layer(ch_in=embed_dim * 32, ch_out=embed_dim * 16)
        # self.Up_Conv5 = UnetBasicBlock(in_dim=embed_dim * 32, out_dim=embed_dim * 16, kernel_size=3, stride=1, padding=1)
        self.Up_Conv5 = DualResBlock(in_dim=embed_dim * 32, out_dim=embed_dim * 16)
        # self.up_fuse5 = ConvBlock(in_channels=embed_dim * 32, out_channels=embed_dim * 16)
        self.up_fuse5 = DWConv(dim=embed_dim * 32, out_dim=embed_dim * 16)

        self.Up4 = up_conv(ch_in=embed_dim * 16, ch_out=embed_dim * 8)
        self.Up_trans4 = _make_bot_layer(ch_in=embed_dim * 16, ch_out=embed_dim * 8)
        # self.Up_Conv4 = UnetBasicBlock(in_dim=embed_dim * 16, out_dim=embed_dim * 8, kernel_size=3, stride=1, padding=1)
        self.Up_Conv4 = DualResBlock(in_dim=embed_dim * 16, out_dim=embed_dim * 8)
        # self.up_fuse4 = ConvBlock(in_channels=embed_dim * 16, out_channels=embed_dim * 8)
        self.up_fuse4 = DWConv(dim=embed_dim * 16, out_dim=embed_dim * 8)

        self.Up3 = up_conv(ch_in=embed_dim * 8, ch_out=embed_dim * 4)
        self.Up_trans3 = _make_bot_layer(ch_in=embed_dim * 8, ch_out=embed_dim * 4)
        # self.Up_Conv3 = UnetBasicBlock(in_dim=embed_dim * 8, out_dim=embed_dim * 4, kernel_size=3, stride=1, padding=1)
        self.Up_Conv3 = DualResBlock(in_dim=embed_dim * 8, out_dim=embed_dim * 4)
        # self.up_fuse3 = ConvBlock(in_channels=embed_dim * 8, out_channels=embed_dim * 4)
        self.up_fuse3 = DWConv(dim=embed_dim * 8, out_dim=embed_dim * 4)

        self.Up2 = up_conv(ch_in=embed_dim * 4, ch_out=embed_dim * 2)
        self.Up_trans2 = _make_bot_layer(ch_in=embed_dim * 4, ch_out=embed_dim * 2)
        # self.Up_conv2 = UnetBasicBlock(in_dim=embed_dim * 4, out_dim=embed_dim * 2, kernel_size=3, stride=1, padding=1)
        self.Up_Conv2 = DualResBlock(in_dim=embed_dim * 4, out_dim=embed_dim * 2)
        # self.up_fuse2 = ConvBlock(in_channels=embed_dim * 4, out_channels=embed_dim * 2)
        self.up_fuse2 = DWConv(dim=embed_dim * 4, out_dim=embed_dim * 2)

        self.cam = CAM()

        self.adown1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embed_dim * 4, embed_dim * 2, 3, 1, 1),
        )
        self.bdown1 = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(embed_dim * 8, embed_dim * 4, 3, 1, 1),
            nn.Conv2d(embed_dim * 4, embed_dim * 2, 3, 1, 1),
        )

        self.adown2 = nn.Conv2d(embed_dim * 2, embed_dim * 4, 3, 2, 1)
        self.bdown2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embed_dim * 8, embed_dim * 4, 3, 1, 1),
        )

        self.adown3 = nn.Conv2d(embed_dim * 4, embed_dim * 8, 3, 2, 1)
        self.bdown3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embed_dim * 16, embed_dim * 8, 3, 1, 1),
        )

        self.adown4 = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim * 8, 3, 2, 1),
            nn.Conv2d(embed_dim * 8, embed_dim * 16, 3, 2, 1),
        )
        self.bdown4 = nn.Conv2d(embed_dim * 8, embed_dim * 16, 3, 2, 1)

        self.Conv_1x1 = nn.Conv2d(embed_dim * 2, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        start_time = time.time()
        x1_1 = self.trans1(x)
        x1_2 = self.Conv1(x)  # 1,64,256,256
        x1 = torch.cat([x1_1, x1_2], dim=1)
        x1 = self.fuse1(x1)  # skip1

        x2 = self.Maxpool(x1)  # 1,64,128,128
        x2_1 = self.trans2(x2)
        x2_2 = self.Conv2(x2)  # 1,128,128,128
        x2 = torch.cat([x2_1, x2_2], dim=1)
        x2 = self.fuse2(x2)  # skip2

        x3 = self.Maxpool(x2)  # 1,128,64,64
        x3_1 = self.trans3(x3)  # 1,256,64,64
        x3_2 = self.Conv3(x3)
        x3 = torch.cat([x3_1, x3_2], dim=1)
        x3 = self.fuse3(x3)  # skip3

        x4 = self.Maxpool(x3)  # 1,256,32,32
        x4_1 = self.trans4(x4)  # 1,512,32,32
        x4_2 = self.Conv4(x4)
        x4 = torch.cat([x4_1, x4_2], dim=1)
        x4 = self.fuse4(x4)  # skip

        x5 = self.Maxpool(x4)  # 1,512,16,16
        x5_1 = self.trans5(x5)  # 1,1024,16,16
        x5_2 = self.Conv5(x5)
        x5 = torch.cat([x5_1, x5_2], dim=1)
        x5 = self.fuse5(x5)

        adown1 = self.adown1(x2)
        bdown1 = self.bdown1(x3)
        skip1 = self.cam(x1, adown1, bdown1)  # 1,64,256,256

        adown2 = self.adown2(x1)
        bdown2 = self.bdown2(x3)
        skip2 = self.cam(x2, adown2, bdown2)  # 1,128,128,128

        adown3 = self.adown3(x2)
        bdown3 = self.bdown3(x4)
        skip3 = self.cam(x3, adown3, bdown3)  # 1,256,64,64

        adown4 = self.adown4(x2)
        bdown4 = self.bdown4(x3)
        skip4 = self.cam(x4, adown4, bdown4)  # 1,512,32,32

        # decoding + concat path
        d5 = self.Up5(x5)  # 1,512,32,32
        d5 = torch.cat((skip4, d5), dim=1)  # 1,1024,32,32
        d5_1 = self.Up_trans5(d5)
        d5_2 = self.Up_Conv5(d5)
        d5 = torch.cat([d5_1, d5_2], dim=1)
        d5 = self.up_fuse5(d5)

        d4 = self.Up4(d5)  # 1,256,64,64
        d4 = torch.cat((skip3, d4), dim=1)  # 1,512,32,32
        d4_1 = self.Up_trans4(d4)
        d4_2 = self.Up_Conv4(d4)
        d4 = torch.cat([d4_1, d4_2], dim=1)
        d4 = self.up_fuse4(d4)

        d3 = self.Up3(d4)  # 1,128,128,128
        d3 = torch.cat((skip2, d3), dim=1)
        d3_1 = self.Up_trans3(d3)
        d3_2 = self.Up_Conv3(d3)
        d3 = torch.cat([d3_1, d3_2], dim=1)
        d3 = self.up_fuse3(d3)

        d2 = self.Up2(d3)  # 1,64,256,256
        d2 = torch.cat((skip1, d2), dim=1)
        d2_1 = self.Up_trans2(d2)
        d2_2 = self.Up_Conv2(d2)  # 1,64,256,256
        d2 = torch.cat([d2_1, d2_2], dim=1)
        d2 = self.up_fuse2(d2)

        d1 = self.Conv_1x1(d2)
        end_time = time.time()
        inference_time = end_time - start_time
        print(inference_time)

        return d1

