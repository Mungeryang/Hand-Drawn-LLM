# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
## Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''Pytorch OpenAI GPT-2 model'''

import math
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union, Tuple

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.gpt2.modeling_gpt2 import load_tf_weights_in_gpt2, GPT2LMHeadModel,GPT2MLP, GPT2Attention,GPT2Block,GPT2Model
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from transformers.modeling_utils import PretrainedConfig,SequenceSummary
from transformers.pytorch_utils import Conv1D,find_pruneable_heads_and_indices,prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    logging,
)
from transformers.utils.model_parallel_utils import assert_device_map,get_device_map
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

'''
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...utils.deprecation import deprecate_kwarg
'''

if version.parse(torch.__version__) >= version.parse("1.6"):
    is_amp_available = True
    from torch.cuda.amp import autocast
else:
    is_amp_available = False
    
class ThisGPT2Config(GPT2Config):
    model_type = "smallcap-gpt2"

    def __init__(
        self,
        # 添加缩放因子是本文最重要的Trick
        # 相当于添加交叉注意力缩放因子后，进行交叉注意力计算的时候会同时对隐藏层维度和每个头的维度进行维度缩减，降低参数量的同时减少了内存消耗
        cross_attention_reduce_factor = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cross_attention_reduce_factor = cross_attention_reduce_factor

class ThisGPTAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)
        
        self.cross_attention_reduce_factor = config.cross_attention_reduce_factor
        
        if self.is_cross_attention: # cross-attention
            # 作者假设，解码器在查询编码器时并不需要那么多丰富的信息，只需要一个更低维度的“摘要”即可
            self.c_attn = Conv1D(int(2 / self.cross_attention_reduce_factor * self.embed_dim),self.embed_dim) # 同时对K V进行维度缩放
            self.q_attn = Conv1D(int(self.embed_dim / self.cross_attention_reduce_factor), self.embed_dim) # 对Q进行维度缩放
            self.c_proj = Conv1D(self.embed_dim, int(self.embed_dim / self.cross_attention_reduce_factor))
        
        else: # self-attention
            self.c_attn = Conv1D(3*self.embed_dim,self.embed_dim)
            self.c_proj = Conv1D(self.embed_dim,self.embed_dim)
    
    # 核心目标是实现 Scaled Dot-Product Attention
    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with torch.amp.autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            # 使用 torch.baddbmm 进行点积计算
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights
    
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # hidden_states为当前模块正在处理的序列
        # encoder_hidden_states:另一个模块（编码器）已经处理完毕并传递过来的序列
        if encoder_hidden_states is not None: # 表示编码器已经处理完并传入了K和V
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            
            ## 关键修改点 - Trick
            # 在原本GPT的基础之上再将 split_size 和 head_dim 缩小
            # GPT2源码中：self.split_size = self.embed_dim；self.head_dim = self.embed_dim // self.num_heads
            split_size = int(self.split_size / self.cross_attention_reduce_factor)
            head_dim = int(self.head_dim / self.cross_attention_reduce_factor)
            
            query = self.q_attn(hidden_states)
            key,value = self.c_attn(encoder_hidden_states).split(split_size,dim=2)
            attention_mask = encoder_attention_mask
            
            ## 关键修改点 
            query = self._split_heads(query,self.num_heads,head_dim)
            key = self._split_heads(key, self.num_heads, head_dim)
            value = self._split_heads(value, self.num_heads, head_dim)
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
            
            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)
        
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        
        if use_cache is True:
            present = (key, value)
        else:
            present = None
        
        if self.reorder_and_upcast_attn:
            # 计算 Attention
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query, key, value, attention_mask, head_mask
            ) 
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        
        attn_output = self._merge_heads(attn_output, self.num_heads, int(self.head_dim / self.cross_attention_reduce_factor))
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
            
        
'''
class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.nx = nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)
'''

# MLP层对应于 GPT-2架构中的 Feed Forward Network - FFN
class ThisGPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        # 获取模型的核心维度
        embed_dim = config.hidden_size
        # 第一个全连接层
        self.c_fc = Conv1D(intermediate_size,embed_dim)
        # 第二个全连接层
        self.c_proj = Conv1D(embed_dim,intermediate_size)
        # 激活函数
        self.act = ACT2FN[config.activation_function] # activation_function="gelu_new"
        # Dropout
        self.dropout = nn.Dropout(config.resid_pdrop)
        
    def forward(self,hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        # 输入 hidden_states (例如: [batch_size, seq_len, 768])
        
        # 通过第一个全连接层，进行维度扩展
        hidden_states = self.c_fc(hidden_states)
        # 应用非线性激活函数
        hidden_states = self.act(hidden_states)
        # 通过第二个全连接层，将维度收缩回原始大小
        hidden_states = self.c_proj(hidden_states)
        # 应用Dropout
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# GPTBlock实现
class ThisGPT2Block(nn.Module):
    def __init__(self, config,layer_index=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        
        self.ln_1 = nn.LayerNorm(hidden_size,eps=config.layer_norm_epsilon)
        self.attn = ThisGPTAttention(config=config,layer_idx=layer_index)
        self.ln_2 = nn.LayerNorm(hidden_size,eps=config.layer_norm_epsilon)
        
        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config=config, is_cross_attention=True, layer_idx=layer_index)
            self.ln_cross_attn = nn.LayerNorm(hidden_size,eps=config.layer_norm_epsilon)

        self.mlp = ThisGPT2MLP(inner_dim, config)
    
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        # 残差赋值
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past = layer_past,
            attention_mask  = attention_mask,
            head_mask = head_mask,
            use_cache = use_cache,
            output_attentions = output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        
        hidden_states = attn_output + residual
        
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,    
            )
            attn_output = cross_attn_outputs[0]
            hidden_states = attn_output + residual
            outputs = outputs + cross_attn_outputs[2:]
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_froward_hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + feed_froward_hidden_states
        
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs
    

class SmallCapGPTBlock(ThisGPT2Block):
    def __init__(self, config, layer_index=None):
        super().__init__(config, layer_index)
        hidden_size = config.hidden_size
        
        if config.add_cross_attention:
            self.crossattention = ThisGPTAttention(config,is_cross_attention=True,layer_idx=layer_index)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
         

class SmallCapGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([SmallCapGPTBlock(config,layer_index=i) for i in range(config.num_hidden_layers)])
        
        
class SmallCapGPT2MHModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = SmallCapGPT2Model(config)


