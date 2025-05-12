import torch
import torch.fft as fft
import torch.nn as nn


def HLEM(x, threshold, Pb_h, Pb_l, useHLComp):
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
        mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] *= (1+Pb_l)

    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    return x_filtered
