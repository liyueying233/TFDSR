# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import is_torch_version, logging
from diffusers.models.activations import get_activation
from diffusers.models.attention import AdaGroupNorm
from diffusers.models.attention_processor import Attention, AttnAddedKVProcessor, AttnAddedKVProcessor2_0
from diffusers.models.dual_transformer_2d import DualTransformer2DModel
from diffusers.models.resnet import Downsample2D, FirDownsample2D, FirUpsample2D, KDownsample2D, KUpsample2D, ResnetBlock2D, Upsample2D
from diffusers.models.transformer_2d import Transformer2DModel

from .unet_2d_blocks import UpBlock2D, CrossAttnUpBlock2D
from .freq_models import FSR

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    transformer_layers_per_block=1,
    num_attention_heads=None,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    attention_type="default",
    resnet_skip_time_act=False,
    resnet_out_scale_factor=1.0,
    cross_attention_norm=None,
    attention_head_dim=None,
    upsample_type=None,
    use_image_cross_attention=False
):
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warn(
            f"It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2DFSR(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift
        )
    elif up_block_type == "CrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
        return CrossAttnUpBlock2DFSR(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            use_image_cross_attention=use_image_cross_attention
        )
    
    raise ValueError(f"{up_block_type} does not exist.")


class CrossAttnUpBlock2DFSR(CrossAttnUpBlock2D):
    def __init__(
        self,
        pb1=0.1,
        pb2=0.4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pb1 = pb1
        self.pb2 = pb2

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        image_encoder_hidden_states: Optional[torch.FloatTensor] = None,
        useAPComp: str = "",
        useHLComp: str = "",
    ):  


        if self.use_image_cross_attention:
            for resnet, attn, image_attn in zip(self.resnets, self.attentions, self.image_attentions):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # FSR enhancement
                hidden_states = FSR(res_hidden_states, hidden_states, self.pb1, self.pb2, useAPComp, useHLComp)
                # print(f"FSR used {useAPComp}")
                # hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                    ## for image cross attention
                    hidden_states = image_attn(
                        hidden_states,
                        encoder_hidden_states=image_encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                    ## for image cross attention
                    hidden_states = image_attn(
                        hidden_states,
                        encoder_hidden_states=image_encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
        else:
            for resnet, attn in zip(self.resnets, self.attentions):
                # pop res hidden states
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                # FSR enhancement
                hidden_states = FSR(res_hidden_states, hidden_states, self.pb1, self.pb2, useAPComp, useHLComp)
                # print(f"FSR used {useAPComp}")
                # hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                else:
                    hidden_states = resnet(hidden_states, temb)
                    hidden_states = attn(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=cross_attention_kwargs,
                        attention_mask=attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        return_dict=False,
                    )[0]
                
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class UpBlock2DFSR(UpBlock2D):
    def __init__(
        self,
        pb1=0.1,
        pb2=0.4,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pb1 = pb1
        self.pb2 = pb2

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, useAPComp="", useHLComp="",):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            
            # FSR enhancement
            
            hidden_states = FSR(res_hidden_states, hidden_states, self.pb1, self.pb2, useAPComp, useHLComp)
            # print(f"FSR used {useAPComp}")
            # hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
