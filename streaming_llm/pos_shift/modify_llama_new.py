import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)
import types

__all__ = ["enable_llama_pos_shift_attention"]


def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

try:
    from transformers.cache_utils import Cache
except Exception:
    Cache = tuple

def llama_pos_shift_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    **kwargs,  # ← 追加引数を受け流す（cache_position など）
):
    """
    StreamingLLM 用 LLaMA Attention 差し替え（LLaMA-3.x / HF 4.44+ 対応版）

    重要ポイント:
      - HF 4.44+ の引数 `cache_position` を受け取り、position_ids として利用
      - HF の新 Cache API (`Cache`) をサポート（get_seq_length / update）
      - RoPE は Query と「今回分の新規 Key」にのみ適用
        （Cache 内の過去 Key はすでに回転済みなので再適用しない）
    """
    bsz, q_len, _ = hidden_states.size()

    # === 新APIの追加引数を吸収 ===
    cache_position = kwargs.pop("cache_position", None)
    if position_ids is None and cache_position is not None:
        position_ids = cache_position  # [bsz, q_len] 想定

    if past_key_value is None and "kv_cache" in kwargs:
        past_key_value = kwargs["kv_cache"]

    # === QKV を作成 ===
    # 形状: 
    #  Q: [bsz, n_heads, q_len, head_dim]
    #  K/V: [bsz, n_kv_heads, q_len, head_dim]
    query_states = self.q_proj(hidden_states)
    key_states   = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # === これまでにキャッシュされている長さ（prev_len）を取得 ===
    prev_len = 0
    if past_key_value is not None:
        if isinstance(past_key_value, Cache) and hasattr(past_key_value, "get_seq_length"):
            prev_len = past_key_value.get_seq_length(getattr(self, "layer_idx", 0))  # 層ごとの長さ
        elif isinstance(past_key_value, (list, tuple)) and len(past_key_value) > 0 and past_key_value[0] is not None:
            prev_len = past_key_value[0].shape[-2]

    # === RoPE の適用 ===
    # 1) Query 側: 現在の position_ids（= cache_position）に基づき回転
    #    HF 4.44+ の LlamaRotaryEmbedding は `seq_len` ではなく `position_ids` を受ける
    cos_q, sin_q = self.rotary_emb(query_states, position_ids=position_ids)
    query_states = apply_rotary_pos_emb_single(query_states, cos_q, sin_q, position_ids)

    # 2) Key 側: 「今回分の新規 Key」だけを回転（過去はCache内で既に回転済み）
    if position_ids is not None:
        key_pos_ids = position_ids + prev_len  # 直近分の位置は過去長をオフセット
    else:
        # 念のためのフォールバック（通常は position_ids が来る）
        key_pos_ids = torch.arange(prev_len, prev_len + q_len, device=key_states.device).unsqueeze(0)

    cos_k, sin_k = self.rotary_emb(key_states, position_ids=key_pos_ids)
    key_states = apply_rotary_pos_emb_single(key_states, cos_k, sin_k, key_pos_ids)

    # === 過去KVの取得（必要なら結合） ===
    past_k = past_v = None
    if past_key_value is not None and prev_len > 0:
        if isinstance(past_key_value, Cache) and hasattr(past_key_value, "key_cache"):
            layer_idx = getattr(self, "layer_idx", 0)
            past_k = past_key_value.key_cache[layer_idx]    # [bsz, n_kv_heads, prev_len, head_dim]（すでにRoPE適用済）
            past_v = past_key_value.value_cache[layer_idx]  # 同上
        else:
            past_k, past_v = past_key_value  # 旧API: タプル

        if past_k is not None and past_v is not None:
            key_states   = torch.cat([past_k, key_states], dim=2)     # seq 次元で結合
            value_states = torch.cat([past_v, value_states], dim=2)

    # === n_kv_heads → n_heads へ拡張（GQA対応） ===
    key_states   = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # === Attention 計算 ===
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output  = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    # === present_key_value の返却 ===
    # Cache には「今回分 q_len の K/V（拡張前＝n_kv_headsのもの）」を追記する
    present_key_value = None
    if use_cache:
        if isinstance(past_key_value, Cache) and hasattr(past_key_value, "update"):
            layer_idx = getattr(self, "layer_idx", 0)
            # update() が期待する形: [bsz, n_kv_heads, q_len, head_dim]
            # いまの key_states は past と結合＆repeat_kv 済みなので、
            # 追記用には「結合・拡張前」の新規分を使うのが正しい。
            k_new = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v_new = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            # RoPE を新規分に再適用（Cacheに格納するのは回転後のK）
            cos_k2, sin_k2 = self.rotary_emb(k_new, position_ids=key_pos_ids)
            k_new = apply_rotary_pos_emb_single(k_new, cos_k2, sin_k2, key_pos_ids)
            past_key_value.update(k_new, v_new, layer_idx)
            present_key_value = past_key_value
        else:
            # 旧API（タプル）互換：結合後をそのまま格納
            present_key_value = (key_states, value_states)

    if output_attentions:
        return attn_output, attn_weights, present_key_value
    else:
        return attn_output, None, present_key_value


def enable_llama_pos_shift_attention(model):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_llama_pos_shift_attention(
                module,
            )

        if isinstance(module, LlamaAttention):
            model._modules[name].forward = types.MethodType(
                llama_pos_shift_attention_forward, model._modules[name]
            )
