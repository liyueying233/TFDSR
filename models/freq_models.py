import torch
import torch.fft as fft
import torch.nn.functional as F
import torch.nn as nn
import math

from .hlem import HLEM
from .apem import APEM


def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda() 

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    return x_filtered

# 主函数
def FSR(res_hidden_states, hidden_states, Pb1, Pb2, useAPComp="", useHLComp=""):
    if torch.isnan(res_hidden_states).any():
        print("res_hidden_states 有nan")
        res_hidden_states[torch.isnan(res_hidden_states)] = 0
        
    if torch.isnan(hidden_states).any():
        print("hidden_states 有nan")
        hidden_states[torch.isnan(hidden_states)] = 0
        
    # ------------ 0 rename to source code -----------------------
    hs_ = res_hidden_states.float()
    h = hidden_states.float()
    
    # --------------- 1 跳跃特征增强 -----------------------
    if useHLComp != "":
        hs_ = HLEM(hs_, threshold=10, useHLComp=useHLComp, Pb_l=0.9, Pb_h=0.05)
        
    # ---------------- 2 主干特征增强 -----------------
    if useAPComp != "": # 主干+跳跃
        aem = APEM(hs_.shape[1]).to(device=hs_.device, dtype=hs_.dtype)
        hs_ = aem(hs_, useAPComp)


    # ------------------- 3 跳跃连接 -----------------------
    hidden_states = torch.cat([h, hs_], dim=1)

    return hidden_states    