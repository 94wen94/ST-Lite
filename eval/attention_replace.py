from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter

# Flash Attention Import
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Warning: Flash Attention not found, falling back to eager implementation.")

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import logging

logger = logging.get_logger(__name__)

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import BaseModelOutputWithPast

import transformers
import sys
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
sys.path.insert(0, str(UTILS_DIR))

from methods import init_pyramidkv, init_vlcache, init_snapkv, init_st_lite
from ui_tars_utils import smart_resize

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding with Multimodal Sections to query and key tensors.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def unrepeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Undo repeat_kv."""
    batch, num_attention_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    num_key_value_heads = num_attention_heads // n_rep
    hidden_states = hidden_states.reshape(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states[:, :, 0, :, :]

def flash_attention_forward_compressed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    is_causal: bool = True,
    training: bool = False,
):
    """
    Flash Attention wrapper designed for compressed KV Cache.
    """
    query_fa = query.transpose(1, 2).contiguous()
    key_fa = key.transpose(1, 2).contiguous()
    value_fa = value.transpose(1, 2).contiguous()
    
    q_len = query_fa.shape[1]
    kv_len = key_fa.shape[1]
    
    # Determine causality: True for prefill (equal lengths), less relevant for decode (q_len=1)
    use_causal = is_causal and (q_len == kv_len)
    
    attn_output = flash_attn_func(
        query_fa,
        key_fa,
        value_fa,
        dropout_p=dropout if training else 0.0,
        softmax_scale=scaling,
        causal=use_causal,
    )
    
    return attn_output


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    is_causal: bool = True,
    **kwargs,
):
    # Repeat KV if necessary
    if key.shape[1] != query.shape[1]:
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
    else:
        key_states = key
        value_states = value

    # Try using Flash Attention
    use_flash = (
        HAS_FLASH_ATTN 
        and query.is_cuda 
        and query.dtype in [torch.float16, torch.bfloat16]
        and not kwargs.get('output_attentions', False)
    )
    
    if use_flash:
        attn_output = flash_attention_forward_compressed(
            query, key_states, value_states,
            scaling=scaling,
            dropout=dropout,
            is_causal=is_causal,
            training=module.training,
        )
        return attn_output, None
    
    # Fallback to eager implementation
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        if attention_mask.size(-1) < key_states.shape[-2]:
            pad_len = key_states.shape[-2] - attention_mask.size(-1)
            pad_shape = list(attention_mask.shape)
            pad_shape[-1] = pad_len
            padding = attention_mask.new_zeros(pad_shape)
            attention_mask = torch.cat([attention_mask, padding], dim=-1)

        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


############# Qwen2.5-VL ###############

def qwen2_5_vl_vision_attention_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
    
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    if position_embeddings is None:
        logger.warning_once(
            "Using `rotary_pos_emb` is deprecated. Use `position_embeddings` instead."
        )
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    kwargs.pop("attention_mask", None)
    if self.config._attn_implementation == "flash_attention_2":
        attention_interface = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )
    else:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = [
            eager_attention_forward(
                self, q, k, v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output


def qwen2_5_vt_forward_layer(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, n_layer: int, **kwargs) -> torch.Tensor:
    """Forward pass through first n_layer layers of vision transformer (early exit)."""
    hidden_states = self.patch_embed(hidden_states)
    
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    
    for layer_num, blk in enumerate(self.blocks):
        if layer_num >= n_layer:
            break  # Early exit
            
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
            
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    
    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]
    
    return hidden_states

def qwen2_5_vt_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens

        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    return hidden_states    


def get_residual_stream_contrast(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None, n_layer: int = 1):
    pixel_values = pixel_values.type(self.visual.dtype)
    
    # get outputs from the final layer of the visual encoder
    image_embeds_final = self.visual(pixel_values, grid_thw=image_grid_thw)
    
    # get outputs from the early layers of the visual encoder
    image_embeds_early = self.visual.forward_early_exit(pixel_values, grid_thw=image_grid_thw, n_layer=n_layer)
    
    # image_embeds_contrast = image_embeds_final - image_embeds_early
    image_embeds_contrast = image_embeds_early
    
    split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    image_embeds_contrast = torch.split(image_embeds_contrast, split_sizes)
    return image_embeds_contrast

def get_residual_stream_cosine_similarity(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None, n_layer: int = 1):
    """Computes cosine similarity between final and early layer embeddings."""
    pixel_values = pixel_values.type(self.visual.dtype)
    
    image_embeds_final = self.visual(pixel_values, grid_thw=image_grid_thw)
    image_embeds_early = self.visual.forward_early_exit(pixel_values, grid_thw=image_grid_thw, n_layer=n_layer)
    
    cosine_sim = F.cosine_similarity(image_embeds_final, image_embeds_early, dim=-1)
    cosine_sim_scaled = (1 - cosine_sim) / 2
    
    split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    cosine_similarities = torch.split(cosine_sim_scaled, split_sizes)
    
    return cosine_similarities
    
def qwen2_5_vl_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size()
    self.scaling = self.head_dim**-0.5
    
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=None,
        **kwargs,
    )
    
    if attn_weights is not None and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()
        
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    
    return attn_output, attn_weights, past_key_value

def qwen2_5_vl_decoder_layer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
    
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
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    
    if self_attn_weights is not None:
        print(f"self_attn_weights device: {self_attn_weights.device}")
    hidden_states = residual + hidden_states

    # Fully Connected
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


def qwen2_5_vl_text_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
      

########## PyramidKV ##########
        
def qwen2_5_vl_attention_forward_PyramidKV(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size()
    self.scaling = self.head_dim**-0.5
    
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    
    # Reset kv_seq_len if we're in prefilling phase
    if q_len > 1:
        self.kv_seq_len = 0
    
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        init_pyramidkv(self, num_hidden_layers=self.config.num_hidden_layers)
        
        # Compress
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups)
            self.kept_indices = self.kv_cluster.kept_indices
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens=self.kv_seq_len

    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=None,
        **kwargs,
    )

    if attn_weights is not None and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()
        
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value

########## SnapKV ##########
        
def qwen2_5_vl_attention_forward_SnapKV(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size()
    self.scaling = self.head_dim**-0.5
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    
    if q_len > 1:
        self.kv_seq_len = 0
    
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        init_snapkv(self)
        
        # Compress
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups)
            self.kept_indices = self.kv_cluster.kept_indices
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens=self.kv_seq_len

    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=None,
        **kwargs,
    )

    if attn_weights is not None and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()
        
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value


########## ST_Lite (Formerly GUI0KV) ##########
        
def qwen2_5_vl_attention_forward_ST_Lite(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size()
    self.scaling = self.head_dim**-0.5
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]
    
    if q_len > 1:
        self.kv_seq_len = 0
    
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        self.config.vision_start_idx = self.vision_start_idx
        self.config.vision_end_idx = self.vision_end_idx
        
        if hasattr(self, 'token_information_scores'):
            self.config.token_information_scores = self.token_information_scores
        
        init_st_lite(self)
        
        # Compress
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups, hidden_states)
            self.kept_indices = self.kv_cluster.kept_indices
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens=self.kv_seq_len

    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=None,
        **kwargs,
    )

    if attn_weights is not None and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()
        
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value

########### VLCache ##########


def qwen2_5_vl_attention_forward_VLCache(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size()
    self.scaling = self.head_dim**-0.5
    
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    kv_seq_len = key_states.shape[-2]

    if q_len > 1:
        self.kv_seq_len = 0
        
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                if q_len == 1:
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        
        # Prefilling compression
        if key_states.shape[-2] == kv_seq_len and hasattr(self, 'gammas') and self.gammas is not None:
            self.kv_seq_len = kv_seq_len
            if hasattr(self, 'betas') and self.betas is not None:
                max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100)) * self.config.num_hidden_layers
                self.config.layer_budget = (max_capacity_prompt * self.betas).int()
                
                init_vlcache(self, num_hidden_layers=self.config.num_hidden_layers)
                
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(key_states, query_states, value_states, attention_mask, self.num_key_value_groups)
                self.kept_indices = self.kv_cluster.kept_indices
                past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            else:
                print(f"Warning: betas not set for layer {self.layer_idx}, skipping compression")
                self.kv_seq_len += q_len
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens=self.kv_seq_len

    # VLCache special handling: Need attention weights for first prefill
    need_attn_weights = (not hasattr(self, 'gammas') or self.gammas is None)
    
    if need_attn_weights:
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,
            output_attentions=True,
            **kwargs,
        )
    else:
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,
            **kwargs,
        )
    
    # update gammas
    if (not hasattr(self, 'gammas') or self.gammas is None):
        self.gammas = []
        if attn_weights is not None and self.last_vision_indices is not None:
            for batch_idx, (this_attn_weights, this_last_vision_indices) in enumerate(zip(attn_weights, self.last_vision_indices)):
                roi_weights = this_attn_weights[:, this_last_vision_indices:, :]
                
                if attention_mask is not None:
                    mask = attention_mask[batch_idx]
                    while mask.dim() > 2: mask = mask.squeeze(0)
                    roi_mask = mask[this_last_vision_indices:, :] > -1e4
                    
                    gammas = []
                    for head_weights in roi_weights:
                        valid_elements = head_weights[roi_mask]
                        sparsity = 1.0 - (valid_elements > 0.01).float().mean().item() if valid_elements.numel() > 0 else 0.0
                        gammas.append(sparsity)
                    avg_gamma = torch.tensor(gammas).mean().item()
                else:
                    flat_weights = roi_weights.flatten(1)
                    sparsity_per_head = 1.0 - (flat_weights > 0.01).float().mean(dim=1)
                    avg_gamma = sparsity_per_head.mean().item()
                
                self.gammas.append(avg_gamma)
            self.gammas = torch.tensor(self.gammas)
        else:
            self.gammas = torch.zeros(bsz)
    
    if attn_weights is not None and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()
        
    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, past_key_value


def qwen2_5_vl_text_model_forward_VLCache(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.dim() == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    if cache_position[0] == 0:  # pre-filling phase
        gammas = []
        for decoder_layer in self.layers:
            decoder_layer.self_attn.betas = None
            decoder_layer.self_attn.gammas = None
            
            decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            gammas.append(decoder_layer.self_attn.gammas)
        
        gammas = torch.stack(gammas, dim=1)
        Z = (1 - gammas).sum(dim=1)
        betas = (1 - gammas) / Z.unsqueeze(1)
        betas = torch.clamp(betas, min=0.001, max=1.0)
        
        for decoder_layer in self.layers:
            decoder_layer.self_attn.betas = betas[:, decoder_layer.self_attn.layer_idx]
    
    total_budget = 0
    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]
        
        if output_attentions:
            all_self_attns += (layer_outputs[1],)
        
        layer_budget = decoder_layer.self_attn.config.layer_budget[0]
        total_budget += layer_budget
        
    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def qwen2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


########## PyramidKV for Qwen2 ##########

def qwen2_attention_forward_PyramidKV(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    
    if q_len > 1:
        self.kv_seq_len = 0
    
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # Repeat KV states for grouped-query attention
    if hasattr(self, 'num_key_value_groups'):
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        init_pyramidkv(self, num_hidden_layers=self.config.num_hidden_layers)
        
        # Compress
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, 
                self.num_key_value_groups if hasattr(self, 'num_key_value_groups') else 1
            )
            self.kept_indices = self.kv_cluster.kept_indices
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len
    
    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    
    if attn_weights is not None and hasattr(self, 'move_attention_to_cpu') and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()
    
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

########## SnapKV for Qwen2 ##########

def qwen2_attention_forward_SnapKV(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    
    if q_len > 1:
        self.kv_seq_len = 0
    
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    if key_states.shape[1] != query_states.shape[1]:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        init_snapkv(self)
        
        # Compress
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, 
                self.num_key_value_groups if hasattr(self, 'num_key_value_groups') else 1
            )
            self.kept_indices = self.kv_cluster.kept_indices
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len

    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    
    if attn_weights is not None and hasattr(self, 'move_attention_to_cpu') and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()
    
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


########## ST_Lite (Formerly GUI0KV) for Qwen2 ##########

def qwen2_attention_forward_ST_Lite(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    
    if q_len > 1:
        self.kv_seq_len = 0
    
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        self.config.vision_start_idx = self.vision_start_idx
        self.config.vision_end_idx = self.vision_end_idx
        if hasattr(self, 'token_information_scores'):
            self.config.token_information_scores = self.token_information_scores
        
        init_st_lite(self)
        
        # Compress
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, 
                self.num_key_value_groups if hasattr(self, 'num_key_value_groups') else 1,
                hidden_states
            )
            
            if key_states.shape[1] != key_states_compress.shape[1]:
                key_states_compress = unrepeat_kv(key_states_compress, self.num_key_value_groups)
                value_states_compress = unrepeat_kv(value_states_compress, self.num_key_value_groups)
            key_states_compress = key_states_compress.contiguous(); value_states_compress = value_states_compress.contiguous()
            self.kept_indices = self.kv_cluster.kept_indices
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len
    
    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    
    if attn_weights is not None and hasattr(self, 'move_attention_to_cpu') and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()
    
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


########## VLCache for Qwen2 ##########

def qwen2_attention_forward_VLCache(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    
    kv_seq_len = key_states.shape[-2]
    
    if q_len > 1:
        self.kv_seq_len = 0
    
    if past_key_value is not None:
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                if q_len == 1:
                    kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    if hasattr(self, 'num_key_value_groups'):
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        self.config.max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100))
        self.config.window_size = min(self.config.window_size, self.config.max_capacity_prompt - 2)
        
        if key_states.shape[-2] == kv_seq_len and hasattr(self, 'gammas') and self.gammas is not None:
            self.kv_seq_len = kv_seq_len
            if hasattr(self, 'betas') and self.betas is not None:
                max_capacity_prompt = int(kv_seq_len * (self.kv_cache_budget / 100)) * self.config.num_hidden_layers
                self.config.layer_budget = (max_capacity_prompt * self.betas).int()
                
                init_vlcache(self, num_hidden_layers=self.config.num_hidden_layers)
                
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                    key_states, query_states, value_states, attention_mask, 
                    self.num_key_value_groups if hasattr(self, 'num_key_value_groups') else 1
                )
                self.kept_indices = self.kv_cluster.kept_indices
                past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            else:
                print(f"Warning: betas not set for layer {self.layer_idx}, skipping compression")
                self.kv_seq_len += q_len
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        past_key_value._seen_tokens = self.kv_seq_len
    
    need_attn_weights = (not hasattr(self, 'gammas') or self.gammas is None)
    
    if need_attn_weights:
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            output_attentions=True,
            **kwargs,
        )
    else:
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
    
    if (not hasattr(self, 'gammas') or self.gammas is None):
        if attn_weights is not None:
            self.gammas = []
            for batch_idx, (this_attn_weights, this_last_vision_indices) in enumerate(zip(attn_weights, self.last_vision_indices)):
                roi_weights = this_attn_weights[:, this_last_vision_indices:, :]
                
                if attention_mask is not None:
                    mask = attention_mask[batch_idx]
                    while mask.dim() > 2: mask = mask.squeeze(0)
                    roi_mask = mask[this_last_vision_indices:, :] > -1e4
                    
                    gammas = []
                    for head_weights in roi_weights:
                        valid_elements = head_weights[roi_mask]
                        sparsity = 1.0 - (valid_elements > 0.01).float().mean().item() if valid_elements.numel() > 0 else 0.0
                        gammas.append(sparsity)
                    avg_gamma = torch.tensor(gammas).mean().item()
                else:
                    flat_weights = roi_weights.flatten(1)
                    sparsity_per_head = 1.0 - (flat_weights > 0.01).float().mean(dim=1)
                    avg_gamma = sparsity_per_head.mean().item()
                
                self.gammas.append(avg_gamma)
            self.gammas = torch.tensor(self.gammas)
        else:
            self.gammas = torch.zeros(bsz)
    
    if attn_weights is not None and hasattr(self, 'move_attention_to_cpu') and self.move_attention_to_cpu:
        attn_weights = attn_weights.detach().cpu()
    
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights



def qwen2_text_model_forward_VLCache(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    if cache_position[0] == 0:  # pre-filling phase
        gammas = []
        for decoder_layer in self.layers:
            decoder_layer.self_attn.betas = None
            decoder_layer.self_attn.gammas = None
            
            layer_attn_mask = causal_mask_mapping[decoder_layer.attention_type]
            decoder_layer(
                hidden_states,
                attention_mask=layer_attn_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            gammas.append(decoder_layer.self_attn.gammas)
        
        gammas = torch.stack(gammas, dim=1)
        Z = (1 - gammas).sum(dim=1)
        betas = (1 - gammas) / Z.unsqueeze(1)
        betas = torch.clamp(betas, min=0.001, max=1.0)
        
        for decoder_layer in self.layers:
            decoder_layer.self_attn.betas = betas[:, decoder_layer.self_attn.layer_idx]
    
    total_budget = 0
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        layer_budget = decoder_layer.self_attn.config.layer_budget[0]
        total_budget += layer_budget
    
    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )

    
def safe_reset_cache(past_key_value):
    """Safely reset cache in a way that's compatible with accelerate"""
    if past_key_value is None:
        return
    
    try:
        if hasattr(past_key_value, 'crop'):
            past_key_value.crop(0)
            past_key_value._seen_tokens = 0
            return
        
        if hasattr(past_key_value, 'key_cache') and hasattr(past_key_value, 'value_cache'):
            past_key_value.key_cache.clear()
            past_key_value.value_cache.clear()
            past_key_value._seen_tokens = 0
            return
            
        past_key_value._seen_tokens = 0
        
    except Exception as e:
        print(f"Warning: Could not fully reset cache: {e}")
        try:
            past_key_value._seen_tokens = 0
        except:
            pass

def disable_accelerate_hooks_for_vlcache(model):
    """Disable accelerate hooks that interfere with VLCache cache management"""
    try:
        hooks_removed = 0
        def remove_hooks(module):
            nonlocal hooks_removed
            if hasattr(module, '_hf_hook') and module._hf_hook is not None:
                module._hf_hook = None
                hooks_removed += 1
                print(f"Removed accelerate hook from {module.__class__.__name__}")
            
            for child in module.children():
                remove_hooks(child)
        
        remove_hooks(model)
        print(f"Removed {hooks_removed} accelerate hooks for VLCache compatibility")
        
    except Exception as e:
        print(f"Error removing accelerate hooks: {e}")

def configure_accelerate_skip_attention(model):
    """Configure accelerate to skip moving attention tensors back to GPU"""
    try:
        hooks_configured = 0
        def configure_hooks(module):
            nonlocal hooks_configured
            if hasattr(module, '_hf_hook') and module._hf_hook is not None:
                if hasattr(module._hf_hook, 'skip_keys'):
                    existing_skip_keys = module._hf_hook.skip_keys
                    
                    if existing_skip_keys is None:
                        skip_keys = {'attentions'}
                    elif isinstance(existing_skip_keys, str):
                        skip_keys = {existing_skip_keys, 'attentions'}
                    elif isinstance(existing_skip_keys, (list, tuple)):
                        skip_keys = set(existing_skip_keys).union({'attentions'})
                    elif isinstance(existing_skip_keys, set):
                        skip_keys = existing_skip_keys.union({'attentions'})
                    else:
                        try:
                            skip_keys = set(existing_skip_keys).union({'attentions'})
                        except:
                            skip_keys = {'attentions'}
                    
                    module._hf_hook.skip_keys = skip_keys
                    hooks_configured += 1
            
            for child in module.children():
                configure_hooks(child)
        
        configure_hooks(model)
        
        if hasattr(model, 'model') and hasattr(model.model, 'forward'):
            original_forward = model.model.forward
            
            def patched_forward(*args, **kwargs):
                outputs = original_forward(*args, **kwargs)
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    cpu_attentions = tuple(
                        attn.cpu() if attn is not None and attn.device.type == 'cuda' else attn 
                        for attn in outputs.attentions
                    )
                    outputs = type(outputs)(
                        **{k: v if k != 'attentions' else cpu_attentions for k, v in outputs.items()}
                    )
                return outputs
            
            model.model.forward = patched_forward
            
    except Exception as e:
        print(f"Error configuring accelerate skip keys: {e}")

def set_attention_implementation(model, args):
    if "UI-TARS" in args.model_path:
        for block in model.model.visual.blocks:
            block.attn._attn_implementation = args.attention_implementation
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.config._attn_implementation = args.attention_implementation
    else:
        raise NotImplementedError(f"Model not supported: {args.model_path}")
    
def set_move_attention_to_cpu(model, args):
    if "UI-TARS" in args.model_path:
        for layer in model.model.language_model.layers:
            layer_name = layer.__class__.__name__
            if args.do_visualization or args.do_attention_sparsity_analysis:
                layer.self_attn.move_attention_to_cpu = True
            else:
                layer.self_attn.move_attention_to_cpu = False
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer_name = layer.__class__.__name__
            if args.do_visualization or args.do_attention_sparsity_analysis:
                layer.self_attn.move_attention_to_cpu = True
            else:
                layer.self_attn.move_attention_to_cpu = False
    else:
        raise NotImplementedError(f"Model not supported: {args.model_path}")
        

def set_kv_cache_budget(model, args):
    if "UI-TARS" in args.model_path:
        for layer in model.model.language_model.layers:
            layer.self_attn.kv_cache_budget = args.kv_cache_budget
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.kv_cache_budget = args.kv_cache_budget
    else:
        raise NotImplementedError(f"Model not supported: {args.model_path}")

def setup_vlcache_compatibility(model, disable_accelerate_hooks=True):
    if disable_accelerate_hooks:
        disable_accelerate_hooks_for_vlcache(model)
    else:
        configure_accelerate_skip_attention(model)

def set_last_vision_indices(model, last_vision_indices, args):
    if "UI-TARS" in args.model_path:
        for layer in model.model.language_model.layers:
            layer.self_attn.last_vision_indices = last_vision_indices
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.last_vision_indices = last_vision_indices
    else:
        raise NotImplementedError(f"Model not supported: {args.model_path}")
        
def set_token_information_scores(model, token_information_scores, args):
    if "UI-TARS" in args.model_path:
        for layer in model.model.language_model.layers:
            layer.self_attn.token_information_scores = token_information_scores
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.token_information_scores = token_information_scores
    else:
        raise NotImplementedError(f"Model not supported: {args.model_path}")
        
def set_vision_start_idx(model, vision_start_idx, args):
    if "UI-TARS" in args.model_path:
        for layer in model.model.language_model.layers:
            layer.self_attn.vision_start_idx = vision_start_idx
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.vision_start_idx = vision_start_idx
    else:
        raise NotImplementedError(f"Model not supported: {args.model_path}")

def set_vision_end_idx(model, vision_end_idx, args):
    if "UI-TARS" in args.model_path:
        for layer in model.model.language_model.layers:
            layer.self_attn.vision_end_idx = vision_end_idx
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.vision_end_idx = vision_end_idx
    else:
        raise NotImplementedError(f"Model not supported: {args.model_path}")

def set_window_size(model, args):
    if "UI-TARS" in args.model_path:
        for layer in model.model.language_model.layers:
            layer.self_attn.config.window_size = args.window_size
    elif "OpenCUA" in args.model_path:
        for layer in model.language_model.model.layers:
            layer.self_attn.config.window_size = args.window_size
    else:
        raise NotImplementedError(f"Model not supported: {args.model_path}")
        
def compute_token_information_scores(image, patch_size=28, factor=28, min_pixels=None, max_pixels=None, temperature=1.5, hidden_states=None, cosine_similarities=None):
    """
    Compute token information scores using cosine similarities or L2 norm of hidden states.
    Falls back to uniform scores if neither is provided.
    """
    if isinstance(image, str):
        image = Image.open(image)
    
    if isinstance(image, Image.Image):
        image_array = np.array(image)
    else:
        image_array = image
    
    if len(image_array.shape) == 3:
        orig_height, orig_width, _ = image_array.shape
    else:
        orig_height, orig_width = image_array.shape
    
    resized_height, resized_width = smart_resize(
        orig_height, orig_width, 
        factor=factor, 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )
    
    num_patches_h = resized_height // patch_size
    num_patches_w = resized_width // patch_size
    
    # If cosine similarities are provided, use them directly
    if cosine_similarities is not None:
        token_information_scores = torch.softmax(cosine_similarities / temperature, dim=0)
        return token_information_scores.float()
    
    # If hidden states are provided, use L2 norm approach
    elif hidden_states is not None:
        importance_scores = torch.norm(hidden_states, dim=-1)
        importance_scores = (importance_scores - importance_scores.mean()) / (importance_scores.std() + 1e-8)
        token_information_scores = torch.softmax(importance_scores / temperature, dim=0)
        return token_information_scores.float()
    
    # Fallback: Return uniform scores
    total_patches = num_patches_h * num_patches_w
    if total_patches == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    
    token_information_scores = torch.ones(total_patches, dtype=torch.float32) / total_patches
    return token_information_scores 


def replace_qwen2_5_vl(kv_cache_mode="full_cache", disable_accelerate_for_vlcache=False):
    
    assert kv_cache_mode in ["full_cache", "pyramid_kv", "vl_cache", "snap_kv", "st_lite"]
    
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention.forward = qwen2_5_vl_vision_attention_forward    
    if kv_cache_mode == "full_cache":
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_5_vl_attention_forward
    elif kv_cache_mode == "pyramid_kv":
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_5_vl_attention_forward_PyramidKV
    elif kv_cache_mode == "snap_kv":
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_5_vl_attention_forward_SnapKV
    elif kv_cache_mode == "st_lite":
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_5_vl_attention_forward_ST_Lite
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VisionTransformerPretrainedModel.forward_early_exit = qwen2_5_vt_forward_layer
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_residual_stream_contrast = get_residual_stream_contrast
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration.get_residual_stream_cosine_similarity = get_residual_stream_cosine_similarity
    elif kv_cache_mode == "vl_cache":
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLAttention.forward = qwen2_5_vl_attention_forward_VLCache
        transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLTextModel.forward = qwen2_5_vl_text_model_forward_VLCache
        
        if disable_accelerate_for_vlcache:
            print("Warning: VLCache mode with accelerate hooks disabled. This may affect memory management.")
            
            
            
def replace_opencua(kv_cache_mode="full_cache", disable_accelerate_for_vlcache=False):
    
    assert kv_cache_mode in ["full_cache", "pyramid_kv", "vl_cache", "snap_kv", "st_lite"]
    
    if kv_cache_mode == "full_cache":
        pass
        # transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attention_forward
    elif kv_cache_mode == "pyramid_kv":
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attention_forward_PyramidKV
    elif kv_cache_mode == "snap_kv":
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attention_forward_SnapKV
    elif kv_cache_mode == "st_lite":
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attention_forward_ST_Lite
    elif kv_cache_mode == "vl_cache":
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = qwen2_attention_forward_VLCache
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = qwen2_text_model_forward_VLCache
        
        if disable_accelerate_for_vlcache:
            print("Warning: VLCache mode with accelerate hooks disabled for OpenCUA. This may affect memory management.")
