# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
# mungeryang 2025.11.25
"""PyTorch ColPali."""

from typing import ClassVar, Optional

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from .paligemma import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaPreTrainedModel,
)

'''
    colpali架构浅谈:
        眼睛 (Vision Tower): SigLIP (输出视觉特征)
        桥梁 (Projector 1): PaliGemmaMultiModalProjector -> self.linear = nn.Linear(config.vision_config.hidden_size 1152, config.vision_config.projection_dim 2048, bias=True)
        大脑 (LLM): Gemma (理解语义)
        压缩 (Projector 2): ColPali 独有的用于将Gemma输出压缩到 128 维，便于存储检索 self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size 2048,self.dim 128)
'''


class Colpali(PaliGemmaPreTrainedModel):
    """
    本质上讲 ColPali 将一个 “生成下一个词的概率模型” 改造成了一个 “输出稠密向量序列的回归/表示模型”
    
    ColPali model implementation from the "ColPali: Efficient Document Retrieval with Vision Language Models" paper.

    Args:
        config (PaliGemmaConfig): The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore all tokens embeddings
            except those of the image at inference.
            Defaults to False --> Do not mask any embeddings during forward pass.
    """
    
    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related
    _checkpoint_conversion_mapping = {
        "^model.language_model.model": "model.model.language_model",
        "^model.vision_tower": "model.model.vision_tower",
        "^model.multi_modal_projector": "model.model.multi_modal_projector",
        "^model.language_model.lm_head": "model.lm_head",
    }
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = cls._checkpoint_conversion_mapping
        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)
    
    def __init__(self, config: PaliGemmaConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config)
        
        # 继承自PaliGemmaForConditionalGeneration
        model = PaliGemmaForConditionalGeneration(config)
        if model.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"model.language_model.{k}" for k in model.language_model._tied_weights_keys]
        self.model = model
        # 检索不需要生成文字 - 这是一个“占位符”层，不做任何数学运算；输入是什么，输出就是什么
        self.model.lm_head = torch.nn.Identity()
        
        # TODO: Wait for ColPali2 to create a ColPaliConfig to allow specifying the embedding dimension.
        # We could do it now but it would break all the models trying to load the model from the checkpoint.
        self.dim = 128
        # Gemma的隐藏层维度为2048，每个Token存为2048维索引太大
        # 增加一个线性层，把2048维度压缩至128维度，保留语义的同时大幅度降低存储成本
        # 它不需要重新预训练 Vision Tower 或 LLM,只需要训练这个 2048 -> 128 的线性层
        self.custom_text_proj = nn.Linear(self.model.config.text_config.hidden_size,self.dim)
        
        # 建立文档索引时，只保留图像部分的 Embeddings,进一步节省存储空间
        self.mask_non_image_embeddings = mask_non_image_embeddings
        self.post_init()
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        # 1. 强制要求内部模型返回 Hidden States
        outputs = self.model(*args, output_hidden_states=True, **kwargs)
        
        # 2. 截取最后一层 Hidden States (Batch, Seq_Len, 2048)
        # 注意：这里不再关心 logits，因为 lm_head 已经是 Identity 了，直接取 hidden_states
        last_hidden_states = outputs.hidden_states[-1]
        
        # 3. 投影 (Batch, Seq_Len, 128)
        proj = self.custom_text_proj(last_hidden_states)
        
        # 4. L2 归一化 (关键步骤！)
        # ColBERT/ColPali 计算相似度用的是点积，保持数值稳定性
        proj = proj / proj.norm(dim=-1,keepdim=True)
        
        # 5. Mask 掉 Padding
        # PAD 的 Token不应该参与检索计算，将其置为 0
        # [[1, 1, 0]] -》 [[[1],[1],[0]]]
        proj = proj * kwargs["attention_mask"].unsqueeze(-1) # unsqueeze(-1) (升维)
        
        return proj

