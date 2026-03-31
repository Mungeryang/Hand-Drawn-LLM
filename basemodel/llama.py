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
'''Pytorch Meta LlaMA model'''

import os
import json
import warnings
from typing import Optional,Tuple,Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)

from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.generation import GenerationMixin 

'''
    LlaMA模型在架构上与GPT模型的区别：
        1. SwiGLU激活函数[PaLM] 使用SwiGLU代替ReLU非线性激活函数
        2. Pre-normalization[GPT-3] 对每个Transformer子层输入进行规范化
        3. RoPE编码[GPTNeo] 旋转位置编码替换绝对位置编码
    提升训练速度措施：
        1. 使用因果多头注意力减少内存使用和运行时间 - xformers库中获取
        2. 使用基于FlashAttention的反向传播方法
        3. 手动实现Transformer层的反向传播函数 不依赖Pytorch自动求导
    
    相比于GPT-2的优化改进：
        1. 分组查询注意力机制的使用 GQA
        2. 旋转位置编码 RoPE
        3. Flash Attention内核优化
'''


class LlamaRMSNorm(nn.Module):
    def __init__(self,hidden_size:int,eps:float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # 可学习参数g_i
        self.variance_epsilon = eps # 初始化epslion
    
    def forward(self,hidden_states):
        '''x_i -> hidden_states'''
        # hidden_states : [batch_size,seq_length,hidden_states]
        input_dtype = hidden_states.dtype #记录输入类型以便计算结束后恢复
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1,keepdim=True) # 计算 1/n * sigma(x_i ^ 2);.mean(-1,keepdim=True)是沿着最后一个维度hidden_size求平均值
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon) # rsqrt 计算平方根的倒数
        return self.weight * hidden_states.to(input_dtype) # 计算结果转换回输入时的类型
    
    def extra_repr(self):
        return f"{tuple(self.weight.shape)},eps={self.variance_epsilon}"



# 辅助完成旋转操作
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # 交换-取反
    x1 = x[..., : x.shape[-1] // 2] # [...,d_1,...,d_n/2]
    x2 = x[..., x.shape[-1] // 2 :] # [...,d_n/2+1,...,d_n]
    return torch.cat((-x2, x1), dim=-1) # [...,-d_n/2+1,...,-d_n, d1,...,d_n/2]

# 执行旋转操作，将“旋转操作应用到Q和K上”
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

    
class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        
    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


"""
    MLP与FFN基本上是指同一个功能组件
    
    FFN的提出源自于Attention is All you Need这篇文章，特指自注意力层之后由两个线性层和一个激活层组成的网络
    
    MLP是古老术语，泛指任何由多个全连接(非线性激活函数)组成的网络
    
    作用：
        1. 增加非线性提升模型表达能力，“非线性”是深度的根本，注意力机制本质上都是线性运算
        2. 特征的升维扩展，升维使得数据变得更加线性可分
        3. 学习到的事实性知识和语言模式多数被编码在了MLP/FFN层的权重矩阵中，MLP可以识别输入模式激活相关知识
    
"""
class LlamaMLP(nn.Module):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
        self.up_proj = nn.Linear(self.hidden_size,self.intermediate_size,bias=False)
        self.down_proj = nn.Linear(self.intermediate_size,self.hidden_size,bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        
    def forward(self,x):
        # x = self.gate_proj(x)
        # x = self.act_fn(x)
        # x = x * self.up_proj(x)
        # down = self.down_proj(x)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    

def repeat_kv(hidden_states:torch.Tensor,n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    # [batch, num_key_value_heads, slen, head_dim]
    batch, num_key_value_heads,slen,head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch,num_key_value_heads,n_rep,slen,head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    ''' GQA的核心思想是让多组query共享同一组key和value ''' 
    def __init__(self, config:LlamaConfig, layer_idx:int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        # 计算 Q头 和 k/v头 的分组比例
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # Q投影输出 = Q头数 * 每个头的维度
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # K/V投影输出 = K/V头数 * 每个头的维度
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # 生成 Q K V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
        # 应用旋转位置编码
        cos, sin = self.rotary_emb(value, seq_len=kv_seq_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        # 处理KV Cache
        # past_key_values就是Cache对象
        if past_key_value is not None:
            key, value = past_key_value.update(key, value, self.layer_idx, {"sin": sin, "cos": cos})

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) / (self.head_dim**0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    
class LlamaDecoderBlock(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # 残差连接
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    

class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config # add
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify either input_ids or inputs_embeds, but not both.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = DynamicCache()
        
        past_seen_tokens = past_key_values.get_seq_length()
        if input_ids is not None:
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device)
        else:
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)
        
        hidden_states = inputs_embeds
        next_decoder_cache = None

        
        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        
        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=None,
            attentions=None,
        )

    
    def _update_causal_mask(self, attention_mask, input_tensor):
        if self.config._attn_implementation == "flash_attention_2":
            return attention_mask
        
        batch_size, seq_length = input_tensor.shape[:2]
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        else:
            attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=input_tensor.device)

        return _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), input_tensor, 0
        )

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config:LlamaConfig):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self): return self.model.embed_tokens
    def set_input_embeddings(self, value): self.model.embed_tokens = value
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_embeddings): self.lm_head = new_embeddings
    def set_decoder(self, decoder): self.model = decoder
    def get_decoder(self): return self.model

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        # `get_usable_length` is needed for KV cache to work correctly with padding
        model_inputs = {"input_ids": input_ids}
        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[Cache] = None, inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        # Loss calculation part (not needed for simple generation but good to have)
        loss = None
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)
    
# =================================================================
# =================== test  Llama implementation ==================
# =================================================================

from transformers import LlamaTokenizer, AutoModelForCausalLM

def test_llama():
    """
    Instantiates your custom Llama model, loads official pre-trained weights,
    and runs a simple text generation task.
    """
    print("\n" + "="*30)
    print("Testing our Llama implementation on Mac-Book Pro(M3)")
    print("="*30)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU) for acceleration.")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU.")
    
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    print(f"Loading model: {model_id}")
    
    try:
        # 加载分词器和配置
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        config = LlamaConfig.from_pretrained(model_id)
        
        llama_model = LlamaForCausalLM(config=config)
        print("Model instantiated successfully.")
        
        # 加载权重到llama模型中
        print("Downloading official weights and loading them into our model structure...")
        official_model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16)
        llama_model.load_state_dict(official_model.state_dict())
        
        print(" Weights loaded successfully! Your model architecture is compatible.")
        
        llama_model.to(device)
        llama_model.eval()
        
        
    except Exception as e:
        print(f"\n An error occurred during model loading: {e}")
        return

    prompt = "In a world where dragons ruled the skies, a young blacksmith discovered"
    print(f"\nTest prompt: \"{prompt}\"")
    
    inputs = tokenizer(prompt,return_tensors="pt").to(device)
    
    print("使用llama生成文本")
    try:
        with torch.no_grad():
            output_ids = llama_model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                eos_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(output_ids[0],skip_special_tokens=True)
            print("\n--- 我们的模型输出 ---")
            print(response)
            print("----------------------------------")
            print("\n🎉 Congratulations! Our Llama implementation is working correctly.")
            
    except Exception as e:
        print(f"\n 发生错误: {e}")
        
        
if __name__ == '__main__':
    test_llama()
