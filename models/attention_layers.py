import torch
import torch.nn as nn

from .freq_utils import fftSplitAP, fftCombineAP

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if torch.isinf(x).any():
            print("selayer刚开始就inf")
        if torch.isnan(x).any():
            print("selayer刚开始就nan")
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        #Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


# Training_AEM_attn_Pa 对于筛选出来的通道做基于attention的增强
class attn_aem(nn.Module):
    def __init__(self, channel, reduction=16, Pa=0.3, Ps=0.3):
        super(attn_aem, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.Pa = Pa
        self.Ps = Ps

    def forward(self, x):
        if torch.isinf(x).any():
            print("selayer刚开始就inf")
        if torch.isnan(x).any():
            print("selayer刚开始就nan")
        
        # attention map
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        # normalization
        y_min = y.min(dim=1, keepdim=True)[0]
        y_max = y.max(dim=1, keepdim=True)[0]
        y = (y - y_min) / (y_max - y_min + 1e-7)
        
        # choose channels
        channel_contributions = torch.abs(x.mean(dim=[2, 3]))  # 形状 [B, 1280]
        c_split = torch.min(channel_contributions) + self.Ps * (torch.max(channel_contributions) - torch.min(channel_contributions))
        small_channel_mask = (channel_contributions <= c_split).unsqueeze(-1).unsqueeze(-1)
        small_channel_mask = small_channel_mask.expand(-1, -1, x.size(2), x.size(3))
        
        # enhance x
        x = torch.where(small_channel_mask, x * (1 - self.Pa) * y, x)
        return x
    

# AEM_channel_hid_NO 对于筛选出来的通道用归一化attn map增强
class attn_aem_chHid(nn.Module):
    def __init__(self, channel, reduction=16, Pa=0.3, Ps=0.3):
        super(attn_aem_chHid, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.Pa = Pa
        self.Ps = Ps
    def forward(self, x):
        if torch.isinf(x).any():
            print("selayer刚开始就inf")
        if torch.isnan(x).any():
            print("selayer刚开始就nan")
        # attention map
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # choose channels
        channel_contributions = torch.abs(x.mean(dim=[2, 3]))  # 形状 [B, 1280]
        c_split = torch.min(channel_contributions) + self.Ps * (torch.max(channel_contributions) - torch.min(channel_contributions))
        small_channel_mask = (channel_contributions <= c_split).unsqueeze(-1).unsqueeze(-1)
        small_channel_mask = small_channel_mask.expand(-1, -1, x.size(2), x.size(3))

        # chHid normalization
        attn_map = ((x * y).mean(3).unsqueeze(3)).mean(2).unsqueeze(2)
        map_max, _ = torch.max(attn_map, dim=-1, keepdim=True) 
        map_min, _ = torch.min(attn_map, dim=-1, keepdim=True)
        attn_map = (attn_map - map_min) / (map_max - map_min + 1e-7)
        
        # enhance x
        x = torch.where(small_channel_mask, x * attn_map, x)
        return x

# IJCAI2024 直接对h进行处理的CSNet 
# https://github.com/c-yn/CSNet/blob/main/Dehazing/ITS/models/CSNet.py
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.PReLU())
        self.main = nn.Sequential(*layers)
    def forward(self, x):
        return self.main(x)
    
class ChannelAttention(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv1 = BasicConv(dim*2, dim*2, kernel_size=1, stride=1)
    def forward(self, x):
        res = x.clone()
        channel_fft = torch.fft.fft2(x, norm='forward', dim=1)
        cat = torch.cat((channel_fft.real, channel_fft.imag), dim=1)
        cat = self.conv1(cat)
        real, imag = torch.chunk(cat, 2, dim=1)
        out = torch.fft.ifft2(torch.complex(real.float(), imag.float()), norm='forward', dim=1)
        return torch.abs(out) + res
    
# 参考ijcai把实部虚部分开，改写成AP
class ChannelAttention_AP(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv1 = BasicConv(dim*2, dim*2, kernel_size=1, stride=1)
    def forward(self, x):
        res = x.clone()
        channel_fft = torch.fft.fft2(x, norm='forward', dim=1)
        cat = torch.cat((torch.abs(channel_fft), torch.angle(channel_fft)), dim=1)
        cat = self.conv1(cat)
        amp, pha = torch.chunk(cat, 2, dim=1)
        out = torch.fft.ifft2(torch.polar(amp.float(), pha.float()), norm='forward', dim=1)
        return torch.abs(out) + res
    
class ChannelAttention_AP_v2(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv1 = BasicConv(dim*2, dim*2, kernel_size=1, stride=1)
    def forward(self, x):
        res = x.clone()
        amp, pha = fftSplitAP(x)
        cat = torch.cat((amp, pha), dim=1)
        cat = self.conv1(cat)
        amp, pha = torch.chunk(cat, 2, dim=1)
        out = fftCombineAP(amp, pha)
        return torch.abs(out) + res

# MICAI2018 CSNET对A_bone处理
# https://github.com/iMED-Lab/CS-Net/blob/master/model/csnet.py
class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels, size):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//8,kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(in_channels//8),
            # nn.LayerNorm([in_channels // 8, size, size]),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(in_channels//8),
            # nn.LayerNorm([in_channels // 8, size, size]),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
    
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        if not torch.isnan(proj_query).any() and not torch.isnan(proj_key).any():
            affinity = torch.matmul(proj_query, proj_key)
            affinity = self.softmax(affinity)
            proj_value = self.value(x).view(B, -1, H * W)
        
            weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
            weights = weights.view(B, C, H, W)
            out = self.gamma * weights + x
        else:
            out = x

        return out
    

# 改写mi csnet（加入aem split）
class MI_CSnet(nn.Module):
    def __init__(self, in_channels, Pa=0.3, Ps=0.3):
        super(MI_CSnet, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//8,kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.Ps = Ps
    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        # choose channels
        channel_contributions = torch.abs(x.mean(dim=[2, 3]))  # 形状 [B, 1280]
        c_split = torch.min(channel_contributions) + self.Ps * (torch.max(channel_contributions) - torch.min(channel_contributions))
        small_channel_mask = (channel_contributions <= c_split).unsqueeze(-1).unsqueeze(-1)
        small_channel_mask = small_channel_mask.expand(-1, -1, x.size(2), x.size(3))
        # enhance x（筛选小通道）
        x = torch.where(small_channel_mask, out, x)
        return x