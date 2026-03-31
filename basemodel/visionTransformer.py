# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
"""Pytorch ViT model"""
import collections.abc
import math
from collections.abc import Callable
from typing import Optional, Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import dataloader,Dataset

import transformers
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedLMOutput,
    MaskedImageModelingOutput
)
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel,ALL_ATTENTION_FUNCTIONS
from transformers.models.vit.configuration_vit import ViTConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

logger = logging.get_logger(__name__)

"""
    Vision Transformer的动机:将NLP的研究范式迁移到CV领域中
    研究证明了对CNN(卷积神经网络)的依赖并不是必要的,直接应用于图像块序列的纯Transformer在图像分类任务可以表现出SOTA的性能
"""

# image - patch分割与embedding
class ViTPatchEmbedding(nn.Module):
    def __init__(self, config:ViTConfig):
        super().__init__()
        
        image_size, patch_size = config.image_size, config.patch_size
        num_channels,hidden_size = config.num_channels, config.hidden_size
        
        image_size = image_size if isinstance(image_size,collections.abc.Iterable) else (image_size,image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings

# image - 加入位置编码的embedding
class ViTEmbedding(nn.Module):
    def __init__(self, config:ViTConfig,use_mask_token: bool = False):
        super().__init__()
        
        self.cls_token = nn.Parameter(torch.randn(1,1,config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1,1,config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbedding(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embedding = nn.Parameter(torch.randn(1,num_patches + 1,config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config
        
    def interpolate_pos_encoding(self,embeddings:torch.Tensor,height:int,width:int) -> torch.Tensor:
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embedding.shape[1] - 1
        
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding

        class_pos_embed = self.position_embedding[:,:1]
        patch_pos_embed = self.position_embedding[:,1:]
        
        dim = embeddings.shape[-1]
        
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        
        sqrt_num_positions = torch.int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1,sqrt_num_positions,sqrt_num_positions,dim)
        # (1,dim,sqrt_num_positions,sqrt_num_positions)
        patch_pos_embed = patch_pos_embed.permute(0,3,1,2)
        
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )
        
        patch_pos_embed = patch_pos_embed.permute(0,2,3,1).view(1,-1,dim)
        return torch.cat((class_pos_embed,patch_pos_embed),dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        
        batch_size,num_channels,height,width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values,interpolate_pos_encoding=interpolate_pos_encoding)
        
        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask
        
        cls_tokens = self.cls_token.expand(batch_size,-1,-1)
        embeddings = torch.cat((cls_tokens,embeddings),dim=1)
        
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings,height,width)
        else:
            embeddings = embeddings + self.position_embedding
            
        embeddings = self.dropout(embeddings)

        return embeddings
        
# 点积注意力的计算      
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
):
    if scaling is None:
        # sqrt(d_k)
        scaling = query.size(-1) ** -0.5
        
        # Q * K.T / sqrt(d_k)
        attn_weights = torch.matmul(query,key.transpose(2,3)) * scaling
        
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key.shape[-2]]
            # 引入位置编码信息
            attn_weights = attn_weights + attention_mask
        # softmax(Q * K.T / sqrt(d_k))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        
        # softmax(Q * K.T / sqrt(d_k)) * value
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1,2).contiguous()
        
        return attn_output,attn_weights
    

"""
    分块出来的思想：
        将原始的输入的hidden_size按照注意力头的数量进行分块处理
""" 
class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.config = config
        # 注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 全部注意力头的数量
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 投影层
        self.dropout_prob = config.attention_probs_dropout_prob
        # 缩放因子
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

        # Q K V映射生成
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size

        # 生成Q K V
        key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)

        # self-attention计算
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            None,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer, attention_probs
    
            
class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config:ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size,config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
        
# 完整的注意力Block
class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self_attn_output, _ = self.attention(hidden_states)
        output = self.output(self_attn_output, hidden_states)
        return output

# 多层感知机 - 非线性激活
class ViTACT(nn.Module):
    def __init__(self, config:ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size,config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN(config.hidden_act)
        else:
            self.act_fn = config.hidden_act
    
    def forward(self,hidden_states:torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        return hidden_states

class VitOutPut(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size,config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class ViTLayer(nn.Module):
    def __init__(self, config:ViTConfig):
        super().__init__()
        self.chunk_size_ffn = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.act = ViTACT(config)
        self.output = VitOutPut(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size,eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm)
        
        hidden_states = attention_output + hidden_states
        
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.act(layer_output)
        
        layer_output = self.output(layer_output,hidden_states)
        return layer_output

class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
    
    def forward(self, hidden_states: torch.Tensor) -> BaseModelOutput:
        for i,layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states)
        
        return BaseModelOutput(last_hidden_state=hidden_states)
    

class ViTPreTrainedModel(PreTrainedModel):
    config: ViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    input_modalities = "image"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ViTEmbeddings", "ViTLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": ViTLayer,
        "attentions": ViTSelfAttention,
    }

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTEmbedding):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)

            if module.mask_token is not None:
                module.mask_token.data.zero_()
                

class ViTPooler(nn.Module):
    """
        ViTPooler 的作用是为整个 Vision Transformer 模型提供一个用于分类任务的接口
        通过提取并处理 [CLS]标记的最终输出来实现
    """
    def __init__(self, config: ViTConfig):
        super().__init__()
        # 线性层将隐藏层大小的向量映射到指定的池化输出大小
        self.dense = nn.Linear(config.hidden_size, config.pooler_output_size)
        # 激活层
        self.activation = ACT2FN[config.pooler_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding to the first token.
        # 提取 [CLS] 标记的最终隐藏状态 - 第一个Token
        # 这个Token正是 [CLS] 标记所有Transformer block后的最终状态
        first_token_tensor = hidden_states[:, 0]
        # 通过一个全连接的隐藏层
        pooled_output = self.dense(first_token_tensor)
        # 通过一个激活函数
        pooled_output = self.activation(pooled_output)
        return pooled_output
                
class ViTModel(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        r"""
        add_pooling_layer (bool, *optional*, defaults to `True`):
            Whether to add a pooling layer
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether to use a mask token for masked image modeling.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbedding(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ViTPatchEmbedding:
        return self.embeddings.patch_embeddings

    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        
    ) -> BaseModelOutputWithPooling:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        # Z_0 = [x_class;x_p1E;x_p2E;···;x_pnE] + E_pos
        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        # Z_l' = MSA(LN(Z_l-1)) + Z_l-1
        encoder_outputs: BaseModelOutput = self.encoder(embedding_output)

        sequence_output = encoder_outputs.last_hidden_state
        # Z_l = MLP(LN(Z_l')) + LN(Z_l')
        sequence_output = self.layernorm(sequence_output)
        # y = LN(Z_L0)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output)
    
        
