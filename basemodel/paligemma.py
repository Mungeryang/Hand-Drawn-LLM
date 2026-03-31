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

"""PyTorch PaliGemmamodel."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_masks_for_generate
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    ModelOutput,
    TransformersKwargs,
    logging,
)

from transformers.models.auto import AutoModel,AutoConfig,CONFIG_MAPPING
# from transformers.models.paligemma.configuration_paligemma import PaliGemmaConfig

logger = logging.get_logger(__name__)

class PaliGemmaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PaliGemmaForConditionalGeneration`]. It is used to instantiate an
    PaliGemmamodel according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the PaliGemma-2B.

    e.g. [paligemma-hf/paligemma-2b](https://huggingface.co/paligemma-hf/paligemma-2b)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vision_config (`PaliGemmaVisionConfig`,  *optional*):
            Custom vision config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        image_token_index (`int`, *optional*, defaults to 256000):
            The image token index to encode the image prompt.
        vocab_size (`int`, *optional*, defaults to 257152):
            Vocabulary size of the PaliGemmamodel. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~PaliGemmaForConditionalGeneration`]
        projection_dim (`int`, *optional*, defaults to 2048):
            Dimension of the multimodal projection space.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden layer of the Language model.

    Example:

    ```python
    >>> from transformers import PaliGemmaForConditionalGeneration, PaliGemmaConfig, SiglipVisionConfig, GemmaConfig

    >>> # Initializing a Siglip-like vision config
    >>> vision_config = SiglipVisionConfig()

    >>> # Initializing a PaliGemma config
    >>> text_config = GemmaConfig()

    >>> # Initializing a PaliGemma paligemma-3b-224 style configuration
    >>> configuration = PaliGemmaConfig(vision_config, text_config)

    >>> # Initializing a model from the paligemma-3b-224 style configuration
    >>> model = PaliGemmaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "paligemma"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False

        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = vision_config.get("model_type", "siglip_vision_model")
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                intermediate_size=4096,
                hidden_size=1152,
                patch_size=14,
                image_size=224,
                num_hidden_layers=27,
                num_attention_heads=16,
                vocab_size=257152,
                vision_use_head=False,
            )

        self.text_config = text_config
        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "gemma")
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["gemma"](
                hidden_size=2048,
                num_hidden_layers=18,
                intermediate_size=16384,
                num_attention_heads=8,
                num_key_value_heads=1,
                is_encoder_decoder=False,
                vocab_size=vocab_size,
            )

        # BC: `use_bidirectional_attention` was originally unset in PaliGemma1 (backbone = Gemma1) AND PaliGemma2
        # (backbone = Gemma2). Both PaliGemmas want to default to True.
        if self.text_config.use_bidirectional_attention is None:
            self.text_config.use_bidirectional_attention = True

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim
        super().__init__(**kwargs)


__all__ = ["PaliGemmaConfig"]


class PaligemmaModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: Optional[torch.FloatTensor] = None

class PaliGemmaCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    
# PaliGemma多模态投影器
# 模态对齐 (让 LLM 看懂图片)	
# 将hidden_states进行线性变换：vision_config.hidden_size -> vision_config.projection_dim
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        hidden_states = self.linear(image_features)

        return hidden_states

# PaliGemma 最精彩的部分
def token_type_ids_mask_function(
    token_type_ids: Optional[torch.Tensor],
    image_group_ids: Optional[torch.Tensor],
) -> Optional[Callable]:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """
    # Do not return an additional mask in this case
    if token_type_ids is None:
        return None
    
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # If it's 1 for both query and key/value, we are in an image block
        # NOTE: static cache shape goes beyond input seq length, while token_type_ids.shape[1] == input seq length
        # Since vmap doesn't support `if statement` we workaround it with `torch.where`
        safe_q_idx = torch.where(q_idx < token_type_ids.shape[1],q_idx,0)
        safe_kv_idx = torch.where(kv_idx < token_type_ids.shape[1], kv_idx, 0)
        
        token_type_ids_at_q_idx = token_type_ids[batch_idx, safe_q_idx]
        token_type_ids_at_q_idx = torch.where(q_idx < token_type_ids.shape[1], token_type_ids_at_q_idx, 0)
        
        token_type_ids_at_kv_idx = token_type_ids[batch_idx, safe_kv_idx]
        token_type_ids_at_kv_idx = torch.where(kv_idx < token_type_ids.shape[1], token_type_ids_at_kv_idx, 0)
        
        image_group_ids_at_q_idx = image_group_ids[batch_idx, safe_q_idx]
        image_group_ids_at_q_idx = torch.where(q_idx < image_group_ids.shape[1], image_group_ids_at_q_idx, -1)
        
        image_group_ids_at_kv_idx = image_group_ids[batch_idx, safe_kv_idx]
        image_group_ids_at_kv_idx = torch.where(kv_idx < image_group_ids.shape[1], image_group_ids_at_kv_idx, -1)
        
        # Query 和 Key 都是图像 Token 吗？
        is_image_block = (token_type_ids_at_q_idx == 1) & (token_type_ids_at_kv_idx == 1)
        # 它们属于同一张图片吗？
        same_image_block = image_group_ids_at_q_idx == image_group_ids_at_kv_idx
        
        # 如果是同一张图内部的两个 Patch，允许互相“看见” (True)
        return is_image_block & same_image_block

    return inner_mask

def create_causal_mask_mapping(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor],
    token_type_ids: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    is_training: bool = False,
    **kwargs,
) -> dict:
    """
    Overwrites the base `create_masks_for_generate` with `token_type_ids` masking to create the causal mask mapping
    for all kinds of forward passes. Paligemma uses a bidirectional mask on the prompt tokens.

    Uses `pixel_values` as an optional input to disambiguate edge cases.
    """
    if is_training and token_type_ids is None:
        raise ValueError("`token_type_ids` is required as a model input when training")
    
    mask_kwargs = {
        "config": config.get_text_config(),
        "input_embeds": input_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    # NOTE: this `is_prompt` logic is not flawless, it fails when we're using a cache eagerly initialized
    # (e.g. compiled prefill) AND `pixel_values` are not provided (i.e. the image data is provided through other
    # means). Determining prefill in that case requires checking data values, which is not compile-compatible.
    maybe_is_prompt = past_key_values is None or not past_key_values.is_initialized or pixel_values is not None
    
    if maybe_is_prompt:
        if token_type_ids is not None:
            # The logic bellow was originally written for Gemma3, where `token_type_ids` is reversed. Let's reverse
            # it to then use exactly the same logic.
            token_type_ids = 1 - token_type_ids
        else:
            logger.warning_once(
                "The input may be the prompt, but `token_type_ids` is not provided. We recommend "
                "passing `token_type_ids` to the model to prevent bad attention masking."
            )
            # BC: when NOT training, use bidirectional mask if sequence length > 1. Otherwise, use the default causal
            # mask. This is incorrect in some advanced use cases, hence the warning above.
            # NOTE: this branch can't be reached when training because `token_type_ids` is required as a model input.
            if input_embeds.shape[1] > 1:
                token_type_ids = torch.ones_like(input_embeds)[:, :, 0]
         
    # Logic originally copied from Gemma3. It holds up for Paligemma as well because Paligemma assumes up to one image
    # per prompt AND we reverse `token_type_ids` above. Gemma3 uses a bidirectional mask for images, tagged through
    # `token_type_ids` 1s. 
    # token_type_ids 模型需要知道序列中哪些是图，哪些是文 
    if token_type_ids is not None and maybe_is_prompt:
        # 1 代表图像 Token，0 代表文本 Token
        is_image = (token_type_ids == 1).to(cache_position.device)
        is_previous_image = nn.functional.pad(is_image, (1, 0), value=0)[:, :-1]
        # 区分多张图像 image_group_ids
        new_image_start = is_image & ~is_previous_image # 检测图像开始的边缘
        image_group_ids = torch.cumsum(new_image_start.int(), dim=1) - 1 # 给每张新图 ID + 1
        image_group_ids = torch.where(is_image, image_group_ids, torch.full_like(token_type_ids, -1))
        mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
            token_type_ids.to(cache_position.device), image_group_ids
        )
    
    return create_masks_for_generate(**mask_kwargs)


class PaliGemmaPreTrainedModel(PreTrainedModel):
    config: PaliGemmaConfig
    base_model_prefix = "model"
    input_modalities = ["image", "text"]
    supports_gradient_checkpointing = True
    _no_split_modules = ["PaliGemmaMultiModalProjector"]
    _skip_keys_device_placement = "past_key_values"
    _can_compile_fullgraph = False
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    

class PaliGemmaModel(PaliGemmaPreTrainedModel):
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}
    # we are filtering the logits/labels so we shouldn't divide the loss based on num_items_in_batch
    accepts_loss_kwargs = False
    
    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config=config.vision_config) # siglip_vision_model
        self.multi_model_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        
        language_model = AutoModel.form_config(config=config.text_config)
        self.language_model = language_model
        
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.text_config_dtype = self.config.get_text_config().dtype or self.dtype
        self.post_init()
        
    # Copied from transformers.models.llava.modeling_llava.LlavaModel.get_input_embeddings with Llava->PaliGemma
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    
    # Copied from transformers.models.llava.modeling_llava.LlavaModel.set_input_embeddings with Llava->PaliGemma
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)
    
    # 视觉特征提取与投影
    def get_image_features(self,pixel_values: torch.FloatTensor):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        # SigLIP提取视觉特征 pixel_values → vision_tower 
        image_output = self.vision_tower(pixel_values)
        selected_image_features = image_output.last_hidden_state # 得到图像的视觉特征 image_output.last_hidden_state
        #  特征投影
        image_features = self.multi_model_projector(selected_image_features)
        # 缩放 - 稳定方差
        image_features = image_features / (self.config.text_config.hidden_size**0.5)
        return image_features

    # 找到 input_ids 中等于 image_token_index 的位置
    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
        
        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask
    
    # input_ids 是给分词器用的查字典的索引
    # inputs_embeds 是给神经网络用的向量
    # VLM中图像被编码成了pixel_values向量，这些图像向量不在文本的词表中，没有任何一个整数 ID 能代表它们
    # 必须传入一个混合向量序列：先把文本 ID 查表变成向量 -> 算出图像特征向量 -> 把图像特征塞进文本嵌入中，最终得到混合的inputs_embeds
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, PaligemmaModelOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

        >>> model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma2-3b-mix-224")
        >>> processor = AutoProcessor.from_pretrained("google/paligemma2-3b-mix-224")

        >>> prompt = "Where is the cat standing?"
        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs,)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Where is the cat standing?\nsnow"
        ```"""
        
        # 互斥检查：input_ids 和 inputs_embeds 只能二选一
        # 要么给我原始 ID 我自己查表，要么直接给我混合好的向量，别两个都给，或者两个都不给
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_id >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_id
            # 通过词表映射后的一长串整数
            # "<image><image>... (重复256次) ...<image><bos>Describe this image."
            # 256000 256000 256000 256000 ... (重复256次) ... 256000 64725 56565 152 485 152
            llm_input_ids = input_ids.clone()
            # self.config.image_token_id（图像占位符的 ID，比如 256000）大于 LLM vocab_size
            llm_input_ids[special_image_mask] = 0  # 先把所有的 256000 临时替换成 0，直接把 256000 传给 `self.get_input_embeddings()`
        else:
            llm_input_ids = input_ids

        # 得到输入的 mbedding
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)
        
        
        # 准备位置索引 - KV Cache
        # 在生成文本时，我们需要知道当前生成的 token 是第几个，以便从缓存中读取之前的 keys/values，或者将新的写入正确位置。
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed 不同于 Llama 从 0 开始编码，Gemma 习惯从 1 开始标记位置。
        
        # -----------------------------------------
        # forward开始至以上部分，是大多数VLM forward 方法中一贯的写法，昨天读了Qwen2_VL的代码也基本上大同小异

        # Merge text and images
        # 多模态特征融合
        # 在Colpali的Offline阶段 pixel_values非空，因为会处理文档页面image，得到image_features
        # 在Colpali的Online阶段 pixel_values为空，因为仅处理文本query，没有image_features
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            # 用 get_image_features 计算得到的 image_features 覆盖掉 inputs_embeds中对应占位符的向量
            # 得到混合了图像特征和文本特征的单一序列 inputs_embeds
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            causal_mask_mapping = create_causal_mask_mapping(
                self.config,
                inputs_embeds,
                attention_mask,
                cache_position,
                past_key_values,
                position_ids,
                token_type_ids,
                pixel_values,
                is_training=self.training,
            )

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, # offline阶段是混合了图像特征和文本特征的单一序列 inputs_embeds，online 阶段是纯文本的 Embedding
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return PaligemmaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

class PaliGemmaForConditionalGeneration(PaliGemmaPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    
    def __init__(self, config: PaliGemmaConfig):
        super().__init__(config)
        self.model = PaliGemmaModel(config)
        # Linear(2048, 256000)线性层，把2048维度的特征映射到词表空间，计算softmax来生成文本
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_image_features(self, pixel_values):
        return self.model.get_image_features(pixel_values) 

    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs], 
    ) -> Union[tuple, PaliGemmaCausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return PaliGemmaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # position_ids in Paligemma are 1-indexed
        if model_inputs.get("position_ids") is not None:
            model_inputs["position_ids"] += 1

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs

    
    def create_masks_for_generate(
        config: PretrainedConfig,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        position_ids: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # Uses the overwritten `create_masks_for_generate` with `token_type_ids` masking
        return create_causal_mask_mapping(
            config,
            input_embeds,
            attention_mask,
            cache_position,
            past_key_values,
            position_ids,
            token_type_ids,
            pixel_values=kwargs.get("pixel_values"),
            **{k: v for k, v in kwargs.items() if k != "pixel_values"},
        )
        
        
            
        
    




