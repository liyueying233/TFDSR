import torch
import torch.fft as fft
import torch.nn as nn
'''
class HEM(nn.Module):
    def __init__(self, threshold, useHLComp):
        super(HEM, self).__init__()
        self.threshold = threshold
        self.useHLComp = useHLComp
        self.Pb_h = nn.Parameter(torch.Tensor([0.1]))  # 可训练参数初始化
        self.Pb_l = nn.Parameter(torch.Tensor([0.1]))  # 可训练参数初始化

    def forward(self, x):
        # FFT
        x_freq = fft.fftn(x, dim=(-2, -1))
        x_freq = fft.fftshift(x_freq, dim=(-2, -1))
        B, C, H, W = x_freq.shape
        
        # adapter [0.5, 1.5]
        Hmax, Hmin = 64, 8
        Pb_h = self.Pb_h * ((H - Hmin) / (Hmax - Hmin) + 0.5)
        Pb_l = self.Pb_l * ((H - Hmin) / (Hmax - Hmin) + 0.5)
        
        # 初始化掩膜
        mask = torch.ones((B, C, H, W)).to(x.device)
        crow, ccol = H // 2, W //2
        
        if self.useHLComp == 'H' or self.useHLComp == 'HL':  # 高频增强
            mask[..., :crow - self.threshold, :] *= (1 + Pb_h)
            mask[..., crow + self.threshold:, :] *= (1 + Pb_h)
            mask[..., :, :ccol - self.threshold] *= (1 + Pb_h)
            mask[..., :, ccol + self.threshold:] *= (1 + Pb_h)
        if self.useHLComp == 'L' or self.useHLComp == 'HL':  # 低频增强
            mask[..., crow - self.threshold:crow + self.threshold, ccol - self.threshold:ccol + self.threshold] *= (1 + Pb_l)

        x_freq = x_freq * mask

        # IFFT
        x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
        x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
        
        return x_filtered
'''
# work代码

def High_Enhance_Module(x, threshold, Pb_h, Pb_l, useHLComp):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    B, C, H, W = x_freq.shape
    
    # adapter [0.5, 1.5]
    Hmax, Hmin = 64, 8
    Pb_h *= ((H - Hmin) / (Hmax - Hmin) + 0.5) / 2
    Pb_l *= ((H - Hmin) / (Hmax - Hmin) + 0.5) / 2
    
    # 初始化掩膜
    mask = torch.ones((B, C, H, W)).cuda() 
    crow, ccol = H // 2, W //2
    
    if useHLComp == 'H': # 高频增强
        mask[..., :crow - threshold, :] *= (1+Pb_h)
        mask[..., crow + threshold:, :] *= (1+Pb_h)
        mask[..., :, :ccol - threshold] *= (1+Pb_h)
        mask[..., :, ccol + threshold:] *= (1+Pb_h)
    elif useHLComp == 'L': # 低频增强
        # mask[..., crow - threshold:crow + threshold, :] *= (1+Pb_l)
        # mask[..., :, crow - threshold:crow + threshold] *= (1+Pb_l)
        mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] *= (1+Pb_l)

    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    return x_filtered


# https://github.com/Xiaofeng-life/SFSNiD/blob/master/methods/MyNightDehazing/SFSNiD.py#L252
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
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Frequency_Spectrum_Dynamic_Aggregation_HL(nn.Module):
    def __init__(self, nc):
        super(Frequency_Spectrum_Dynamic_Aggregation_HL, self).__init__()
        self.processhigh_real = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0)
        )
        self.processhigh_imag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0)
        )
        self.processlow_real = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0)
        )
        self.processlow_imag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            SELayer(channel=nc),
            nn.Conv2d(nc, nc, 1, 1, 0)
        )
        
    def forward(self, x, useHLComp, threshold):
        B, C, H, W = x.shape
        low_mask = torch.zeros((B, C, H, W), dtype=x.dtype, device=x.device)
        crow, ccol = H // 2, W // 2
        low_mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = 1
        
        ori_low = low_mask * x
        ori_high = (1 - low_mask) * x

        # 分离实部和虚部
        ori_low_real = ori_low.real
        ori_low_imag = ori_low.imag
        ori_high_real = ori_high.real
        ori_high_imag = ori_high.imag
        
        if useHLComp == 'H':  # 高频增强
            high_real = self.processhigh_real(ori_high_real)
            high_imag = self.processhigh_imag(ori_high_imag)
            high = torch.complex(high_real, high_imag)
            high = ori_high + high
            low = ori_low
        elif useHLComp == 'L':  # 低频增强
            low_real = self.processlow_real(ori_low_real)
            low_imag = self.processlow_imag(ori_low_imag)
            low = torch.complex(low_real, low_imag)
            low = ori_low + low
            high = ori_high
        elif useHLComp == 'HL':  # 高低频增强
            high_real = self.processhigh_real(ori_high_real)
            high_imag = self.processhigh_imag(ori_high_imag)
            high = torch.complex(high_real, high_imag)
            high = ori_high + high
            
            low_real = self.processlow_real(ori_low_real)
            low_imag = self.processlow_imag(ori_low_imag)
            low = torch.complex(low_real, low_imag)
            low = ori_low + low
        else:
            raise ValueError('useHLComp must be H, L, or HL')
        x_out = high + low
        return x_out


# class HEM(nn.Module):
#     def __init__(self, nc):
#         super(HEM, self).__init__()
#         self.fsda = Frequency_Spectrum_Dynamic_Aggregation_HL(nc)

#     def forward(self, x, useHLComp, threshold):
#         _, _, H, W = x.shape
#         x_freq = torch.fft.rfft2(x, norm='backward')
#         x_freq = self.fsda(x_freq, useHLComp, threshold)
#         x = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
#         return x