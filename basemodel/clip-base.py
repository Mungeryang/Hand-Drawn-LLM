# coding=utf-8
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
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
"""PyTorch CLIP model"""

import math
import os
import warnings
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Union, Tuple, Any
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask,_prepare_4d_attention_mask
from transformers.modeling_utils import PreTrainedModel,ALL_ATTENTION_FUNCTIONS

from transformers.models.clip.configuration_clip import CLIPConfig,CLIPTextConfig,CLIPVisionConfig

logger = logging.get_logger(__name__)

'''
    动手实现CLIP模型 2025.09.22
'''


'''
    CLIP的图像编码方式有两种：
        1. 基于ResNet的卷积方法
        2. 基于ViT的方法
'''

# CLIP损失函数(对于没有接触过对比损失的人来说，这是最有趣的部分)
# CLIP的目标是对应图像和文本的向量对齐，意味着点积尽可能接近1，对于其他值需要将其推向0
# 对于给定的标题，对所有图像的点积取softmax ，然后取交叉熵损失；对于给定的图像，对所有标题重复此过程，这两个损失取平均值。
# 希望所有对角线元素都排列整齐(True Positive)，而所有非对角线元素则趋向于零。
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits,torch.arange(len(logits),device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    """
    This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
    model `executorch` exportable. 
    """
    square_tensor = torch.pow(tensor,2)
    sum_tensor = torch.sum(square_tensor,dim=-1,keepdim=True)
    normed_tensor = torch.pow(sum_tensor,0.5)
    return normed_tensor

class CLIPVisionModelOutput(BaseModelOutput):
    r"""
    image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The image embeddings obtained by applying the projection layer to the pooler_output. 将投影层应用到pooler_output获得的图像嵌入。
    """
    iamge_embeds: Optional[torch.FloatTensor] = None # 新增
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class CLIPTextModelOutput(BaseModelOutput):
    r"""
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
        The text embeddings obtained by applying the projection layer to the pooler_output. 投影层应用到pooler_output获得的文本嵌入。
    """
    
    text_embeds: Optional[torch.FloatTensor] = None # 新增
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
class CLIPOutput(BaseModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
        Contrastive loss for image-text similarity. 图像文本相似度的对比损失
        
    logits_per_image (`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
        The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
        similarity scores.  `image_embeds`与`text_embeds`的点积分数，表示 '图像-文本' 对的相似度分数
        
    logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
        The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
        similarity scores. `text_embeds`与`image_embeds`的积分数，表示 '文本-图像' 对的相似度分数
        
    text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
    image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
        The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
    text_model_output (`BaseModelOutputWithPooling`):
        The output of the [`CLIPTextModel`].
    vision_model_output (`BaseModelOutputWithPooling`):
        The output of the [`CLIPVisionModel`].
    """
    loss: Optional[torch.FloatTensor] = None
    logits_per_image: Optional[torch.FloatTensor] = None
    logits_per_text: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None
    
    def to_tuple(self) -> tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self,k).to_tuple()
            for k in self.keys()
        )
    

# CLIP视觉编码器
class CLIPVisionEmbedding(nn.Module):
    def __init__(self, config:CLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions,self.embed_dim)
        self.register_buffer("position_ids",torch.arange(self.num_positions).expand((1,-1)),persistent=False)
    
    # 添加位置编码信息 - 这部分的代码与ViT中 ViTEmbedding 几乎一致
    def interpolate_pos_encoding(self,embeddings:torch.Tensor,height:int,width:int) -> torch.Tensor:
        """
            通过将投影层应用于pooler_output获得的图像嵌入。
            此方法允许对预训练的位置编码进行插值，以便能够在更高分辨率的图像上使用该模型。此方法也适用于支持torch.jit跟踪。
        """
        num_patches = embeddings.shape[1] - 1
        position_embedding = self.position_embedding.weight.unsqueeze(0)
        num_positions = position_embedding.shape[1] - 1
        
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)
        
        class_pos_embed = position_embedding[:, :1]
        patch_pos_embed = position_embedding[:, 1:]
        
        dim = embeddings.shape[-1]
        
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        
        sqrt_num_positions = torch.int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        return torch.cat((class_pos_embed,patch_pos_embed),dim=1)

    def forward(self,pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        if not interpolate_pos_encoding and (height != self.image_size or width != self.image_size):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size}*{self.image_size})."
            )
        
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype)) # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1,2)
        
        class_embeds = self.class_embedding.expand(batch_size,1,-1)
        embeddings = torch.cat([class_embeds,patch_embeds], dim=1)
        
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
            
        return embeddings
    

# CLIP文本编码器
class CLIPTextEmbedding(nn.Module):
    def __init__(self, config:CLIPTextConfig):
        super().__init__()
        self.config = config
        # 嵌入维度
        embed_dim = config.hidden_size
        
        # 序列编码
        self.token_embedding = nn.Embedding(config.vocab_size,embed_dim)
        # 位置编码
        self.position_embedding = nn.Embedding(config.max_position_embeddings,embed_dim)
        
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        max_position_embedding = self.position_embedding.weight.shape[0]
        
        if seq_length > max_position_embedding:
            raise ValueError(
                f"Sequence length must be less than max_position_embeddings (got `sequence length`: "
                f"{seq_length} and max_position_embeddings: {max_position_embedding}"
            )
        
        if position_ids is None:
            position_ids = self.position_ids[:,:seq_length]
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
        
        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings
        
        return embeddings

# 点积缩放注意力计算
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
    attn_weights = torch.matmul(query,key.transpose(-1,-2))*scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    
    attn_weights = nn.functional.softmax(attn_weights,dim=-1,dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    
    attn_output = torch.matmul(attn_weights,value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights

# CLIP的注意力机制
class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    
    def __init__(self, config:Union[CLIPVisionConfig, CLIPTextConfig]):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        self.is_causal = False
        
        self.q_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim,self.embed_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
    )-> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""
        batch_size, seq_len,embed_dim = hidden_states.shape
        
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        query = query.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
        # CLIP text model uses both `causal_attention_mask` and `attention_mask`
        if self.config._attn_implementation == "flash_attention_2":
            self.is_causal = causal_attention_mask is not None
        else:
            if attention_mask is not None and causal_attention_mask is not None:
                attention_mask = attention_mask + causal_attention_mask
            elif causal_attention_mask is not None:
                attention_mask = causal_attention_mask
        
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )
        
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

# CLIP多层感知机
class CLIPMLP(nn.Module):
    def __init__(self, config:Union[CLIPVisionConfig, CLIPTextConfig]):
        super().__init__()
        self.config = config
        self.act_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size,config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size,config.hidden_size)
        
    def forward(self,hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

# 单CLIP编码块
class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
    ) -> torch.FloatTensor:
        
        residual = hidden_states
        
        # 层归一化
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states,attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )
        # 残差连接
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        # 层归一化
        hidden_states = self.layer_norm2(hidden_states)
        # 多层感知机非线形映射
        hidden_states = self.mlp(hidden_states)
        # 残差连接
        hidden_states = residual + hidden_states
        
        return hidden_states

# CLIP 编码器
class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """
    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        super().__init__()
        self.config = config
        # 堆叠多层 CLIPEncoderLayer 构建CLIP编码器
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
    
    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
    ) -> BaseModelOutput:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
        """
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask,
            )
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
        )
        

# 构建CLIP的文本Transformer处理器   
class CLIPTextTransformer(nn.Module):
    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        # 初始化CLIP文本嵌入
        self.embeddings = CLIPTextEmbedding(config)
        # 初始化CLIP编码器
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)
        
        self.eos_token_id = config.eos_token_id
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPooling: 
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        # 进行文本嵌入
        hidden_states = self.embeddings(input_ids=input_ids,position_ids=position_ids)
        
        # 位置编码嵌入
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )
        
        if attention_mask is not None and self.config._attn_implementation != "flash_attention_2":
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        # 编码器输入
        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )
        
        # 得到最终层的输出
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        
        if self.eos_token_id == 2:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]
        
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output
        )

class CLIPTextModel(PreTrainedModel):
    
    config: CLIPTextConfig
    input_modalities = "text"
    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer"]
    _supports_flash_attn = False  # mask creation only accounts for sdpa/eager
    
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        self.text_model = CLIPTextTransformer(config)
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value
    
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, CLIPTextModel

        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        return self.text_model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            position_ids = position_ids
        )



class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        
        self.embeddings = CLIPVisionEmbedding(config)
        self.pre_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.final_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: Optional[bool] = False,
    ) -> BaseModelOutputWithPooling:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layernorm(hidden_states)
        
        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds = hidden_states,
        )
        
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:,0,:]
        pooled_output = self.final_layernorm(pooled_output)
        
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
        )

class CLIPVisionModel(nn.Module):
    config: CLIPVisionConfig
    main_input_name = "pixel_values"
    input_modalities = "image"
    _no_split_modules = ["CLIPEncoderLayer"]
    
    def __init__(self, config:CLIPVisionConfig):
        super().__init__()    
        self.config = config
        self.vision_model = CLIPVisionTransformer(config)
        self.post_init()
        
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> BaseModelOutputWithPooling:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        
        return self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
    
# CLIP模型
class CLIPModel(nn.Module):
    config: CLIPConfig
    _no_split_modules = ["CLIPTextEmbeddings", "CLIPEncoderLayer", "CLIPVisionEmbeddings"]
    def __init__(self, config:CLIPConfig):
        super().__init__()
        
        if not isinstance(config.text_config, CLIPTextConfig):
            raise TypeError(
                "config.text_config is expected to be of type CLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )
        
        text_config = config.text_config
        vision_config  = config.vision_config
        
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        
        text_model = CLIPTextModel._from_config(text_config)
        self.text_model = text_model.text_model
        
        vision_model = CLIPVisionModel._from_config(vision_config)
        self.vision_model = vision_model.vision_model
        
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPTextModel`].

        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, CLIPModel

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> with torch.inference_mode():
        ...     text_features = model.get_text_features(**inputs)
        ```"""
        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        pooled_output = text_outputs.pooler_output
        text_features = self.text_projection(pooled_output)
        
        return text_features

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].

        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, CLIPModel
        >>> from transformers.image_utils import load_image

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = load_image(url)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.inference_mode():
        ...     image_features = model.get_image_features(**inputs)
        ```"""
        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        pooled_output = vision_outputs.pooler_output
        image_features = self.visual_projection(pooled_output)

        return image_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> CLIPOutput:
        r"""
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.

        Examples:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, CLIPModel
        >>> from transformers.image_utils import load_image

        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = load_image(url)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> with torch.inference_mode():
        ...     outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        vision_outputs: BaseModelOutputWithPooling = self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        text_outputs: BaseModelOutputWithPooling = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        
        image_embeds = vision_outputs.pooler_output
        image_embeds = self.visual_projection(image_embeds)
        
        text_embeds = text_outputs.pooler_output
        text_embeds = self.text_projection(text_embeds)
        
        image_embeds = image_embeds / _get_vector_norm(image_embeds)
        text_embeds = text_embeds / _get_vector_norm(text_embeds)
        
        logits_per_text = torch.matmul(text_embeds, image_embeds.t().to(text_embeds.device))
        logits_per_text = logits_per_text * self.logit_scale.exp().to(text_embeds.device)
        
        logits_per_image = logits_per_text.t()
        
        loss  = None
        if return_loss:
            loss = clip_loss(logits_per_text)
        
        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
        
        
        
        
        
        
        
        
        
        



