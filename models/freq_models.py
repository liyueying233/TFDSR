import torch
import torch.fft as fft
import torch.nn.functional as F
import torch.nn as nn
import math

from .attention_layers import ChannelAttention_AP_v2, SELayer, SpatialAttentionBlock, attn_aem, attn_aem_chHid, ChannelAttention, eca_layer, MI_CSnet
from .freq_utils import fftSplitAP, fftCombineAP
from .hem import High_Enhance_Module
from .aem import AEM


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
        hs_ = High_Enhance_Module(hs_, threshold=10, useHLComp=useHLComp, Pb_l=0.9, Pb_h=0.05)
        
    # ---------------- 2 主干特征增强 -----------------
    if useAPComp != "": # 主干+跳跃
        aem = AEM(hs_.shape[1]).to(device=hs_.device, dtype=hs_.dtype)
        hs_ = aem(hs_, useAPComp)


    # ------------------- 3 跳跃连接 -----------------------
    hidden_states = torch.cat([h, hs_], dim=1)

    return hidden_states

# def AEM_attn_mi(x_bone, Pa=0.3, Ps=0.3):
#     """mi attention作为增强的对象"""
#     x_bone = x_bone.float()
#     # Step 1: Perform FFT and split A & P
#     A_bone, P_bone = fftSplitAP(x_bone)
#     # Step 2: choose channels
#     channel_contributions = torch.abs(A_bone.mean(dim=[2, 3]))  # 形状 [B, 1280]
#     c_split = torch.min(channel_contributions) + Ps * (torch.max(channel_contributions) - torch.min(channel_contributions))
#     small_channel_mask = (channel_contributions <= c_split).unsqueeze(-1).unsqueeze(-1)
#     small_channel_mask = small_channel_mask.expand(-1, -1, A_bone.size(2), A_bone.size(3))
#     # Step 3: channel attention layer
#     channel = x_bone.shape[1]
#     h = x_bone.shape[2]
#     attn = SpatialAttentionBlock(channel, h).to(device=x_bone.device, dtype=x_bone.dtype)
#     assert not torch.isnan(A_bone).any()
#     attn_map = attn(A_bone)
#     # Step 4: normalization
#     hidden_mean = attn_map.mean(1).unsqueeze(1)
#     B = hidden_mean.shape[0]
#     hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
#     hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
#     hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
#     assert not torch.isnan(hidden_mean).any()
#     # Step 5: enhance
#     A_bone = torch.where(small_channel_mask, attn_map * (1 - hidden_mean * Pa), attn_map)
#     # Step 6: reconstruct x_bone
#     x_reconstructed = fftCombineAP(A_bone, P_bone)
#     return x_reconstructed

# def Amp_Enhance_Module(x_bone):
#     """AEM_csnet_split 变化不大"""
#     x_bone = x_bone.float()
    
#     # Step 1: Perform FFT and split A & P
#     A_bone, P_bone = fftSplitAP(x_bone)
    
#     # Step 2: channel attention layer
#     channel = x_bone.shape[1]
#     selayer = MI_CSnet(channel).to(device=x_bone.device, dtype=x_bone.dtype)

#     # Step 3: Enhance A_Bone
#     A_bone = selayer(A_bone)
    
#     # Step 4: reconstruct x_bone
#     x_reconstructed = fftCombineAP(A_bone, P_bone)
#     return x_reconstructed

# def Amp_Enhance_Module(x_bone):
#     """AEM_csnet_AP_v2"""
#     x_bone = x_bone.float()
    
#     attn = ChannelAttention_AP_v2(x_bone.shape[1]).to(device=x_bone.device, dtype=x_bone.dtype)
#     x_bone = attn(x_bone)

#     return x_bone

# def Amp_Enhance_Module(x_bone):
#     """AEM_csnet_mi"""
#     x_bone = x_bone.float()
    
#     # Step 1: Perform FFT and split A & P
#     A_bone, P_bone = fftSplitAP(x_bone)
    
#     # Step 2: channel attention layer
#     channel = x_bone.shape[1]
#     selayer = SpatialAttentionBlock(channel).to(device=x_bone.device, dtype=x_bone.dtype)

#     # Step 3: Enhance A_Bone
#     A_bone = selayer(A_bone)
    
#     # Step 4: reconstruct x_bone
#     x_reconstructed = fftCombineAP(A_bone, P_bone)
#     return x_reconstructed

# def Amp_Enhance_Module(x_bone):
#     """AEM_ecalayer"""
#     x_bone = x_bone.float()
    
#     # Step 1: Perform FFT and split A & P
#     A_bone, P_bone = fftSplitAP(x_bone)
    
#     # Step 2: channel attention layer
#     channel = x_bone.shape[1]
#     selayer = eca_layer(channel).to(device=x_bone.device, dtype=x_bone.dtype)

#     # Step 3: Enhance A_Bone
#     A_bone = selayer(A_bone)
    
#     # Step 4: reconstruct x_bone
#     x_reconstructed = fftCombineAP(A_bone, P_bone)
#     return x_reconstructed


# # ==========所有都用/筛选后的结果用过attn的x_A==============
# def Amp_Enhance_Module(x_bone, Pa=0.3, Ps=0.3):
#     """seesr_AEM_part_attn"""
#     x_bone = x_bone.float()
    
#     # Step 1: Perform FFT and split A & P
#     A_bone, P_bone = fftSplitAP(x_bone)
    
#     # Step 2: choose channels
#     channel_contributions = torch.abs(A_bone.mean(dim=[2, 3]))  # 形状 [B, 1280]
#     c_split = torch.min(channel_contributions) + Ps * (torch.max(channel_contributions) - torch.min(channel_contributions))
#     small_channel_mask = (channel_contributions <= c_split).unsqueeze(-1).unsqueeze(-1)
#     small_channel_mask = small_channel_mask.expand(-1, -1, A_bone.size(2), A_bone.size(3))

#     # Attention Map Enhance
#     # Step 3: channel attention layer
#     channel = x_bone.shape[1]
#     selayer = SELayer(channel).to(device=x_bone.device, dtype=x_bone.dtype)
#     attn_map = selayer(A_bone)

#     # 归一化
#     hidden_mean = attn_map.mean(1).unsqueeze(1)
#     B = hidden_mean.shape[0]
#     hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
#     hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
#     hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

#     A_bone = torch.where(small_channel_mask, attn_map * (1 - hidden_mean * Pa), A_bone)
      
#     # Step 4: reconstruct x_bone
#     x_reconstructed = fftCombineAP(A_bone, P_bone)
#     return x_reconstructed




# ================加上Attention：SE==================
# def Amp_Enhance_Module(x_bone):
#     x_bone = x_bone.float()
    
#     # Step 1: Perform FFT and split A & P
#     A_bone, P_bone = fftSplitAP(x_bone)
    
#     # Step 2: channel attention layer
#     channel = x_bone.shape[1]
#     selayer = SELayer(channel).to(device=x_bone.device, dtype=x_bone.dtype)

#     # Step 3: Enhance A_Bone
#     A_bone = selayer(A_bone)
    
#     # Step 4: reconstruct x_bone
#     x_reconstructed = fftCombineAP(A_bone, P_bone)
#     return x_reconstructed


# # 对筛选出来的结果做Attention：SE+AEM
# def Amp_Enhance_Module(x_bone):
#     x_bone = x_bone.float()
#     # Step 1: Perform FFT and split A & P
#     A_bone, P_bone = fftSplitAP(x_bone)

#     # Step 2: channel attention layer
#     channel = x_bone.shape[1]
#     selayer = attn_aem(channel).to(device=x_bone.device, dtype=x_bone.dtype)

#     # Step 3: Enhance A_Bone
#     A_bone = selayer(A_bone)

#     # Step 4: reconstruct x_bone
#     x_reconstructed = fftCombineAP(A_bone, P_bone)
#     return x_reconstructed


# =======直接AEM+Attention层==================
# def AEM_attn(x_bone):
#     """直接将x_bone过channel attention类"""
#     x_bone = x_bone.float()
#     # Step 1: Perform FFT and split A & P
#     A_bone, P_bone = fftSplitAP(x_bone)
#     # Step 2: channel attention layer
#     channel = x_bone.shape[1]
#     selayer = attn_aem_chHid(channel).to(device=x_bone.device, dtype=x_bone.dtype)
#     # Step 3: Enhance A_Bone
#     A_bone = selayer(A_bone)
#     # Step 4: reconstruct x_bone
#     x_reconstructed = fftCombineAP(A_bone, P_bone)
#     return x_reconstructed


# # ==========Attention map后的A_bone作为增强算子操作（涨点）==============
# def Amp_Enhance_Module(x_bone, Pa=0.3, Ps=0.3):
#     """seesr_AEM_a3_attn"""
#     x_bone = x_bone.float()
    
#     # Step 1: Perform FFT and split A & P
#     A_bone, P_bone = fftSplitAP(x_bone)
    
#     # Step 2: choose channels
#     channel_contributions = torch.abs(A_bone.mean(dim=[2, 3]))  # 形状 [B, 1280]
#     c_split = torch.min(channel_contributions) + Ps * (torch.max(channel_contributions) - torch.min(channel_contributions))
#     small_channel_mask = (channel_contributions <= c_split).unsqueeze(-1).unsqueeze(-1)
#     small_channel_mask = small_channel_mask.expand(-1, -1, A_bone.size(2), A_bone.size(3))

#     # Attention Map Enhance
#     # Step 3: channel attention layer
#     channel = x_bone.shape[1]
#     selayer = SELayer(channel).to(device=x_bone.device, dtype=x_bone.dtype)
#     attn_map = selayer(A_bone)

#     # 归一化
#     hidden_mean = attn_map.mean(1).unsqueeze(1)
#     B = hidden_mean.shape[0]
#     hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
#     hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
#     hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

#     A_bone = torch.where(small_channel_mask, A_bone * (1 - hidden_mean * Pa), A_bone)
      
#     # Step 4: reconstruct x_bone
#     x_reconstructed = fftCombineAP(A_bone, P_bone)
#     return x_reconstructed



# =================================原始代码==========================================================================
# def Amp_Enhance_Module(x_bone, Pa=0.3, Ps=0.3):
#     x_bone = x_bone.float()
    
#     # Step 1: Perform FFT and split A & P
#     A_bone, P_bone = fftSplitAP(x_bone)
    
#     # Step 2: choose channels
#     channel_contributions = torch.abs(A_bone.mean(dim=[2, 3]))  # 形状 [B, 1280]
#     c_split = torch.min(channel_contributions) + Ps * (torch.max(channel_contributions) - torch.min(channel_contributions))
#     small_channel_mask = (channel_contributions <= c_split).unsqueeze(-1).unsqueeze(-1)
#     small_channel_mask = small_channel_mask.expand(-1, -1, A_bone.size(2), A_bone.size(3))
    
#     # Step 3: Enhance A_Bone
#     hidden_mean = A_bone.mean(1).unsqueeze(1)
#     B = hidden_mean.shape[0]
#     hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
#     hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
#     hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
#     A_bone = torch.where(small_channel_mask, A_bone * (1 - hidden_mean * Pa), A_bone)
      
#     # Step 4: reconstruct x_bone
#     x_reconstructed = fftCombineAP(A_bone, P_bone)
#     return x_reconstructed






    