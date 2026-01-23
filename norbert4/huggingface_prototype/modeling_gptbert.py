from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import _softmax_backward_data as _softmax_backward_data

from functools import partial, lru_cache

from .configuration_gptbert import GptBertConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.activations import gelu_new
from transformers.utils import is_flash_attn_2_available, logging
from transformers.modeling_outputs import (
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    BaseModelOutput,
    CausalLMOutput
)
import math
from typing import TYPE_CHECKING, Optional, Union, Tuple, List

logger = logging.get_logger(__name__)


# Workaround for transformers < 4.36.0 check_imports issue
# See: https://github.com/huggingface/transformers/issues/28459
try:
    if is_flash_attn_2_available():
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
        from flash_attn.layers.rotary import RotaryEmbedding
        from flash_attn.ops.triton.rotary import apply_rotary
    else:
        flash_attn_varlen_qkvpacked_func, RotaryEmbedding, apply_rotary = None, object, None
        logger.warning_once(
            "NorBERT4 støtter FlashAttention, men det er ikke funnet i miljøet ditt. Du bør vurdere å oppdatere miljøet ditt for å få raskere og mindre minnekrevende behandling."
        )
except ImportError:
    flash_attn_varlen_qkvpacked_func, RotaryEmbedding, apply_rotary = None, object, None
    logger.warning_once(
        "NorBERT4 støtter FlashAttention, men det er ikke funnet i miljøet ditt. Du bør vurdere å oppdatere miljøet ditt for å få raskere og mindre minnekrevende behandling."
    )


# from https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modeling_modernbert.py
@torch.compiler.disable()
def _unpad_input(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    if input_ids.dim() == 2:
        unpadded_inputs = input_ids.flatten()[indices]
    else:
        batch_size, sequence_length, *rest = input_ids.shape
        shape = batch_size * sequence_length
        unpadded_inputs = input_ids.view(shape, *rest)[indices]

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch


# from https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modeling_modernbert.py
def _pad_output(input_ids: torch.Tensor, indices: torch.Tensor, batch_size: int, sequence_length: int) -> torch.Tensor:
    if input_ids.dim() == 1:
        output = torch.zeros(batch_size * sequence_length, dtype=input_ids.dtype, device=input_ids.device)
        output[indices] = input_ids
        padded_inputs = output.view(batch_size, sequence_length)
    else:
        _, *rest = input_ids.shape
        output = torch.zeros(batch_size * sequence_length, *rest, dtype=input_ids.dtype, device=input_ids.device)
        output[indices] = input_ids
        padded_inputs = output.view(batch_size, sequence_length, *rest)

    return padded_inputs


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), bias=self.bias.type_as(x) if self.bias is not None else None)


class CastedLinearIn(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias=bias)
        self.scale = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return F.linear(x, (self.weight * (self.scale + 1.0).unsqueeze(0)).type_as(x), bias=self.bias.type_as(x) if self.bias is not None else None)


class MultiCastedLinearOrthoIn(nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.ParameterList()
        for out_feature in out_features:
            self.weights.append(nn.Parameter(torch.empty((out_feature, in_features))))

        if bias:
            self.bias = nn.Parameter(torch.zeros(sum(out_features)))
        else:
            self.bias = self.register_parameter("bias", None)

        self.scale = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return F.linear(x, (torch.cat([weight for weight in self.weights], dim=0) * (self.scale + 1.0).unsqueeze(0)).type_as(x), bias=self.bias.type_as(x) if self.bias is not None else None)


class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * gelu_new(gate)


class Embedding(nn.Module):
    def __init__(self, config: GptBertConfig):
        super().__init__()

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False, bias=False)
        self.word_scale = nn.Parameter(torch.zeros(config.hidden_size))
        self.dropout = nn.Dropout(config.embedding_dropout)

    def forward(self, input_ids: torch.Tensor):
        word_embedding = self.word_embedding(input_ids)
        word_embedding = self.word_norm(word_embedding)
        word_embedding = word_embedding * (self.word_scale + 1.0)

        return self.dropout(word_embedding)


class LMClassifier(nn.Module):
    def __init__(self, config: GptBertConfig, n_labels: int):
        super().__init__()

        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.projection = CastedLinearIn(config.hidden_size, config.hidden_size, bias=False)
        self.post_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.emb2vocab = CastedLinearIn(config.hidden_size, n_labels, bias=True)

    def forward(self, x: torch.Tensor):
        x = self.pre_norm(x.float()).type_as(x)
        x = self.projection(x)
        x = gelu_new(x)
        x = self.post_norm(x.float()).type_as(x)
        x = self.emb2vocab(x)
        return x


class Classifier(nn.Module):
    def __init__(self, config: GptBertConfig, n_labels: int):
        super().__init__()

        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.projection = CastedLinearIn(config.hidden_size, config.hidden_size, bias=False)
        self.post_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.output_projection = CastedLinearIn(config.hidden_size, n_labels, bias=True)

    def forward(self, x: torch.Tensor):
        x = self.pre_norm(x.float()).type_as(x)
        x = self.projection(x)
        x = gelu_new(x)
        x = self.post_norm(x.float()).type_as(x)
        x = self.dropout(x)
        x = self.output_projection(x)
        return x


# from https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modeling_modernbert.py
def flash_attention_forward(qkv: torch.Tensor, rotary_emb: UnpaddedRotaryEmbedding, cu_seqlens: torch.Tensor, max_seqlen: int, causal: bool, local_attention: Tuple[int, int], dropout_p: float, deterministic: bool, target_dtype: torch.dtype = torch.bfloat16, **_kwargs):
    qkv = rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
    if convert_dtype:
        # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
        # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
        orig_dtype = qkv.dtype
        qkv = qkv.to(target_dtype)

        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=dropout_p,
            deterministic=deterministic,
            window_size=local_attention,
            causal=False
        )
        attn = attn.to(orig_dtype)  # type: ignore
    else:
        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=dropout_p,
            deterministic=deterministic,
            window_size=local_attention,
            causal=False
        )
    return attn


# from https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modeling_modernbert.py
class ApplyRotaryEmbUnpad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, cos, sin, cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None):
        # (total_nnz, 3, nheads, headdim)
        qkv = qkv.contiguous()
        total_nnz, _three, _nheads, headdim = qkv.shape
        # We need qkv to be contiguous so that when we reshape to combine (3, nheads) dimensions,
        # we get the same tensor
        # qk = rearrange(qkv[:, :2], "b_s t h d -> b_s (t h) d")
        qk = qkv[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(qk, cos, sin, seqlen_offsets=0, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, interleaved=False, inplace=True)

        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, do):
        cos, sin, cu_seqlens = ctx.saved_tensors
        do = do.contiguous()
        total_nnz, _three, _nheads, headdim = do.shape
        # We need dqkv to be contiguous so that when we reshape to combine (3, nheads) dimensions,
        # we get the same tensor
        dqk = do[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(
            dqk,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=False,
            inplace=True,
            conjugate=True,
        )

        return do, None, None, None, None, None, None


# from https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modeling_modernbert.py
def apply_rotary_unpadded(qkv, cos, sin, cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None):
    return ApplyRotaryEmbUnpad.apply(qkv, cos, sin, cu_seqlens, max_seqlen)


# from https://github.com/huggingface/transformers/blob/main/src/transformers/models/modernbert/modeling_modernbert.py
class UnpaddedRotaryEmbedding(RotaryEmbedding):
    def __init__(self, dim: int, base: float = 10000.0, max_seqlen: Optional[int] = None):
        super().__init__(dim=dim, base=base, device=None, interleaved=False)
        self.max_seqlen = max_seqlen

    def forward(self, qkv: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: Optional[int] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)

        qkv = apply_rotary_unpadded(
            qkv,
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        return qkv


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, config, theta: int):
        super().__init__()

        head_size = config.query_key_head_size
        assert head_size % 2 == 0
        max_seq_len = config.max_sequence_length

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size))
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        embedding = torch.einsum('n, d -> nd', pos, inv_freq)
        embedding = torch.cat([embedding, embedding], dim=-1).unsqueeze(0)
        self.register_buffer("cos_matrix", embedding.cos(), persistent=False)
        self.register_buffer("sin_matrix", embedding.sin(), persistent=False)

    def forward(self, x: torch.Tensor):
        hidden_layer = x.float()

        seq_len = x.shape[2]

        cos_matrix = self.cos_matrix[:, None, :seq_len, :]
        sin_matrix = self.sin_matrix[:, None, :seq_len, :]

        x_rotate_half = torch.cat(
            [
                -hidden_layer[:, :, :, x.size(-1) // 2:],
                hidden_layer[:, :, :, :x.size(-1) // 2]
            ],
            dim=-1
        )

        out = hidden_layer * cos_matrix + x_rotate_half * sin_matrix
        return out.type_as(x)


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, mask: torch.BoolTensor, dim: int) -> torch.Tensor:
        ctx.dim = dim
        x.masked_fill_(mask, float('-inf'))
        x = torch.softmax(x, ctx.dim)
        x.masked_fill_(mask, 0.0)
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        output: torch.Tensor

        output, = ctx.saved_tensors
        inputGrad: torch.Tensor = _softmax_backward_data(grad_output, output, ctx.dim, output.dtype)
        return inputGrad, None, None


class SelfAttention(nn.Module):
    def __init__(self, config: GptBertConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.d_qk = config.query_key_head_size
        self.d_v = config.value_head_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.q_out_dim = self.d_qk * self.num_attention_heads
        self.k_out_dim = self.d_qk * self.num_kv_heads
        self.v_out_dim = self.d_v * self.num_kv_heads

        self.qk_proj = MultiCastedLinearOrthoIn(self.hidden_size, [self.q_out_dim, self.k_out_dim], bias=False)
        self.v_proj = CastedLinearIn(self.hidden_size, self.v_out_dim, bias=False)
        self.out_proj = CastedLinearIn(self.d_v*self.num_attention_heads, self.hidden_size, bias=False)

        self.pre_v_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.pre_qk_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.inter_norm = nn.LayerNorm(self.d_v * self.num_attention_heads, eps=config.layer_norm_eps, elementwise_affine=False)
        self.q_norm = nn.LayerNorm(self.d_qk, eps=config.layer_norm_eps, elementwise_affine=False, bias=False)
        self.k_norm = nn.LayerNorm(self.d_qk, eps=config.layer_norm_eps, elementwise_affine=False, bias=False)
        self.k_scale = nn.Parameter(torch.ones(self.num_kv_heads, self.d_qk))
        self.q_scale = nn.Parameter(torch.ones(self.num_attention_heads, self.d_qk))

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.dropout = nn.Dropout(config.hidden_dropout)

        theta = 160_000 if (layer_idx + 1) % config.local_global_ratio == 0 else 10_000

        # Initialize rotary embeddings based on whether FlashAttention is available
        if flash_attn_varlen_qkvpacked_func is not None:
            self.rope_embedding = UnpaddedRotaryEmbedding(dim=self.d_qk, base=theta, max_seqlen=config.max_sequence_length)
        else:
            self.rope_embedding = RotaryPositionalEmbeddings(config, theta)

        self.scale = 1.0 / math.sqrt(self.d_qk)
        self.lambdas = nn.Parameter(torch.tensor([0.5]))

        self.sequence_length = config.max_sequence_length
        self.is_causal = config.is_decoder
        self.window_length = None

    def set_window_length(self, window_length: int):
        self.window_length = window_length

    def _get_window_mask(self, query_length: int, key_length: int, device: torch.device):
        """Create and cache window attention mask."""
        if self.is_causal:
            mask = torch.ones(query_length, key_length, dtype=torch.bool, device=device)
            mask = mask.tril().triu(diagonal=-self.window_length)
        else:
            mask = torch.ones(query_length, key_length, dtype=torch.bool, device=device)
            mask = mask.tril(diagonal=self.window_length).triu(diagonal=-self.window_length)
        return mask.view(1, 1, query_length, key_length)

    def attention_operation(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, padding_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard attention computation with masking."""
        batch_size, _, query_length, _ = query.size()
        _, _, key_length, _ = key.size()

        # Use cached window mask
        with torch.no_grad():
            window_mask = self._get_window_mask(query_length, key_length, query.device)
            if padding_mask is not None:
                attention_mask = padding_mask & window_mask
            else:
                attention_mask = window_mask

        attention_scores = torch.bmm(query.flatten(0, 1), key.transpose(-1, -2).flatten(0, 1)) * self.scale  # shape: [B*H, Q_T, K_T]
        attention_scores = attention_scores.view(batch_size, self.num_attention_heads, query_length, key_length)

        attention_probabilities = MaskedSoftmax.apply(attention_scores, ~attention_mask, -1)
        attention_probabilities = self.attention_dropout(attention_probabilities)

        output = torch.bmm(attention_probabilities.flatten(0, 1), value.flatten(0, 1))
        output = output.view(batch_size, self.num_attention_heads, query_length, self.d_v)

        return output

    def forward(self, hidden_layer: torch.Tensor, qk_layer: torch.Tensor, v1: torch.Tensor | None, padding_info):
        # Get original shape info
        if flash_attn_varlen_qkvpacked_func is not None:
            # Unpadded case
            indices, cu_seqlens, max_seqlen = padding_info
            total_seqlen = hidden_layer.size(0)
            batch_size = cu_seqlens.size(0) - 1
        else:
            # Padded case
            batch_size, seq_length = hidden_layer.size(0), hidden_layer.size(1)

        hidden_layer = self.pre_v_norm(hidden_layer.float()).type_as(hidden_layer)
        qk_layer = self.pre_qk_norm(qk_layer.float()).type_as(qk_layer)

        query, key = self.qk_proj(qk_layer).tensor_split([self.q_out_dim], dim=-1)
        value = self.v_proj(hidden_layer)

        if flash_attn_varlen_qkvpacked_func is not None:
            # Reshape for FlashAttention: (total_seqlen, num_heads, head_dim)
            query = query.view(total_seqlen, self.num_attention_heads, self.d_qk)
            key = key.view(total_seqlen, self.num_kv_heads, self.d_qk)
            value = value.view(total_seqlen, self.num_kv_heads, self.d_v)

            # Apply layer norm and scaling
            query = ((self.q_scale + 1.0).unsqueeze(0) * self.q_norm(query.float())).type_as(query)
            key = ((self.k_scale + 1.0).unsqueeze(0) * self.k_norm(key.float())).type_as(key)

            if v1 is None:
                v1 = value
            value = (1 - self.lambdas[0]) * value + self.lambdas[0] * v1

            # Prepare qkv for FlashAttention
            qkv = torch.stack([query, key, value], dim=1)  # (total_seqlen, 3, num_heads, head_dim)

            # Determine window size for local attention
            if self.window_length is not None and self.window_length > 0:
                if self.is_causal:
                    local_attention = (self.window_length - 1, 0)
                else:
                    local_attention = (self.window_length - 1, self.window_length - 1)
            else:
                local_attention = (-1, -1)

            # Apply FlashAttention
            output = flash_attention_forward(
                qkv,
                self.rope_embedding,
                cu_seqlens,
                max_seqlen,
                self.is_causal,
                local_attention,
                self.config.attention_dropout if self.training else 0.0,
                self.config.deterministic_flash_attn
            )

            # Reshape output back
            output = output.view(total_seqlen, self.d_v * self.num_attention_heads)

        else:
            # Standard attention path
            query_length = query.size(1)
            key_length = key.size(1)

            query = query.reshape(batch_size, query_length, self.num_attention_heads, self.d_qk).transpose(1, 2)
            key = key.reshape(batch_size, key_length, self.num_kv_heads, self.d_qk).transpose(1, 2)
            value = value.reshape(batch_size, key_length, self.num_kv_heads, self.d_v).transpose(1, 2)

            query = ((self.q_scale + 1.0).unsqueeze(1).unsqueeze(0) * self.q_norm(query.float())).type_as(query)
            key = ((self.k_scale + 1.0).unsqueeze(1).unsqueeze(0) * self.k_norm(key.float())).type_as(key)

            if v1 is None:
                v1 = value
            else:
                value = (1 - self.lambdas[0]) * value + self.lambdas[0] * v1

            # Apply rotary embeddings
            query = self.rope_embedding(query)
            key = self.rope_embedding(key)

            output = self.attention_operation(query, key, value, padding_info)
            output = output.transpose(1, 2).flatten(2, 3)  # shape: [B, T, H*D]

        output = self.inter_norm(output.float()).type_as(output)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output, v1


class FeedForward(nn.Module):    
    def __init__(self, config: GptBertConfig):
        super().__init__()
        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.up_proj = MultiCastedLinearOrthoIn(config.hidden_size, [config.intermediate_size, config.intermediate_size], bias=False)
        self.activation = GeGLU()
        self.inter_norm = nn.LayerNorm(config.intermediate_size, eps=config.layer_norm_eps, elementwise_affine=False)
        self.down_proj = CastedLinearIn(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout)
        
    def forward(self, x: torch.Tensor):
        x = self.pre_norm(x.float()).type_as(x)
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.inter_norm(x.float()).type_as(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class Layer(nn.Module):
    def __init__(self, config: GptBertConfig, layer_idx: int):
        super().__init__()

        self.attention = SelfAttention(config, layer_idx)
        self.mlp = FeedForward(config)
        self.lambdas = nn.Parameter(torch.tensor([0., 0., 1., 0., 1., 0.]))

    def set_window_length(self, window_length: int):
        self.attention.set_window_length(window_length)

    def forward(self, hidden_layer: torch.Tensor, embeddings: torch.Tensor, v1: torch.Tensor | None, padding_info):
        attention_output = (1 - self.lambdas[0]) * hidden_layer + self.lambdas[0] * embeddings
        qk_layer = (1 - self.lambdas[1]) * hidden_layer + self.lambdas[1] * embeddings
        mlp_layer = F.softplus(self.lambdas[2]) * ((1 - self.lambdas[3]) * hidden_layer + self.lambdas[3] * embeddings)

        attention_output, v1 = self.attention(attention_output, qk_layer, v1, padding_info)
        mlp_layer = mlp_layer + attention_output
        hidden_layer = F.softplus(self.lambdas[4]) * ((1 - self.lambdas[5]) * hidden_layer + self.lambdas[5] * embeddings)
        output = hidden_layer + attention_output + self.mlp(mlp_layer)

        return output, v1


class Encoder(nn.Module):
    def __init__(self, config: GptBertConfig):
        super().__init__()
        self.layers = nn.ModuleList([Layer(config, i) for i in range(config.num_layers)])
        self.local_global_ratio = config.local_global_ratio

    def set_window_length(self, config: GptBertConfig):
        for i, layer in enumerate(self.layers):
            if (i + 1) % self.local_global_ratio == 0:
                layer.set_window_length(config.global_window_length)
            else:
                layer.set_window_length(config.local_window_length)

    def forward(self, hidden_layer: torch.Tensor, padding_info, output_hidden_states=False, checkpoint_activations=False):
        hidden_layers = [hidden_layer] if output_hidden_states else None
        v1 = None
        embeddings = hidden_layer

        for layer in self.layers:
            if checkpoint_activations:
                hidden_layer, v1 = torch.utils.checkpoint.checkpoint(layer, hidden_layer, embeddings, v1, padding_info, use_reentrant=True)
            else:
                hidden_layer, v1 = layer(hidden_layer, embeddings, v1, padding_info)

            if output_hidden_states:
                hidden_layers.append(hidden_layer)

        return hidden_layer, hidden_layers


#
# HuggingFace wrappers
#

class GptBertPreTrainedModel(PreTrainedModel):
    config_class = GptBertConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = False

    def _init_weights(self, module):
        std = math.sqrt(2.0 / (5.0 * self.hidden_size))

        if isinstance(module, nn.Linear) or isinstance(module, CastedLinearIn):
            nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-2*std, b=2*std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight.data, mean=0.0, std=std, a=-2*std, b=2*std)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GptBertModel(GptBertPreTrainedModel):
    def __init__(self, config: GptBertConfig, add_mlm_layer=False, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.hidden_size = config.hidden_size

        self.embedding = Embedding(config)
        self.encoder = Encoder(config)
        self.classifier = LMClassifier(config, config.vocab_size) if add_mlm_layer else None
        self.set_window_length(config)
        self.gradient_checkpointing = False
        self.post_init()

    def set_window_length(self, config) -> None:
        self.encoder.set_window_length(config)

    def get_input_embeddings(self):
        return self.embedding.word_embedding

    def set_input_embeddings(self, value):
        self.embedding.word_embedding = value

    def get_contextualized_embeddings(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            raise ValueError("You have to specify input_ids")

        batch_size, seq_length = input_shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=device)
        else:
            attention_mask = attention_mask.bool()

        if flash_attn_varlen_qkvpacked_func is not None:
            if len(attention_mask.size()) != 2:
                raise ValueError("Bare `attention_mask` med to dimensjoner støttes nå for FlashAttention.")
            with torch.no_grad():
                input_ids, indices, cu_seqlens, max_seqlen_in_batch = _unpad_input(input_ids, attention_mask)
            padding_info = (indices, cu_seqlens, max_seqlen_in_batch)
        else:
            if len(attention_mask.size()) == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif len(attention_mask.size()) == 3:
                attention_mask = attention_mask.unsqueeze(1)
            padding_info = attention_mask

        static_embeddings = self.embedding(input_ids)

        original_dtype = static_embeddings.dtype
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and static_embeddings.dtype == torch.float32:
            static_embeddings = static_embeddings.bfloat16()

        last_layer, contextualized_embeddings = self.encoder(
            static_embeddings,
            padding_info,
            output_hidden_states=output_hidden_states,
            checkpoint_activations=self.gradient_checkpointing and self.training
        )

        last_layer = last_layer.to(original_dtype)
        if output_hidden_states:
            contextualized_embeddings = [layer.to(original_dtype) for layer in contextualized_embeddings]

        # Pad output if using FlashAttention
        if flash_attn_varlen_qkvpacked_func is not None:
            last_layer = _pad_output(last_layer, indices, batch_size, seq_length)
            if output_hidden_states:
                contextualized_embeddings = [_pad_output(layer, indices, batch_size, seq_length) for layer in contextualized_embeddings]
            else:
                contextualized_embeddings = None

        return last_layer, contextualized_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings = self.get_contextualized_embeddings(input_ids, attention_mask, output_hidden_states)

        if not return_dict:
            return (
                sequence_output,
                *([contextualized_embeddings] if output_hidden_states else [])
            )

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=contextualized_embeddings if output_hidden_states else None
        )


class GptBertForMaskedLM(GptBertModel):
    _tied_weights_keys = ["classifier.emb2vocab.weight"]

    def __init__(self, config: GptBertConfig, **kwargs):
        super().__init__(config, add_mlm_layer=True, **kwargs)

    def get_output_embeddings(self):
        return self.classifier.emb2vocab.weight

    def set_output_embeddings(self, new_embeddings):
        self.classifier.emb2vocab.weight = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings = self.get_contextualized_embeddings(input_ids, attention_mask, output_hidden_states)
        subword_prediction = self.classifier(sequence_output)
        subword_prediction = 30 * torch.sigmoid(subword_prediction / 7.5)

        masked_lm_loss = None
        if labels is not None:
            labels_flatten = labels[:, 1:].flatten()
            subword_prediction_flatten = subword_prediction[:, :-1].flatten(0, 1)
            masked_lm_loss = F.cross_entropy(subword_prediction_flatten, labels_flatten)

        bos_logits = torch.zeros(subword_prediction.size(0), 1, self.config.vocab_size, dtype=subword_prediction.dtype, device=subword_prediction.device)
        bos_logits[:, :, self.config.bos_token_id] = 1.0
        subword_prediction = torch.cat([bos_logits, subword_prediction[:, :-1]], dim=1)

        if not return_dict:
            output = (
                subword_prediction,
                *([contextualized_embeddings] if output_hidden_states else [])
            )
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=subword_prediction,
            hidden_states=contextualized_embeddings if output_hidden_states else None
        )


class GptBertForCausalLM(GptBertModel):
    _tied_weights_keys = ["classifier.emb2vocab.weight"]

    def __init__(self, config: GptBertConfig, **kwargs):
        config.is_decoder = True
        super().__init__(config, add_mlm_layer=True, **kwargs)

    def get_output_embeddings(self):
        return self.classifier.emb2vocab.weight

    def set_output_embeddings(self, new_embeddings):
        self.classifier.emb2vocab.weight = new_embeddings

    def get_input_embeddings(self):
        return self.embedding.word_embedding

    def set_input_embeddings(self, value):
        self.embedding.word_embedding = value

    def set_decoder(self, decoder):
        self.encoder = decoder

    def get_decoder(self):
        return self.encoder

    def can_generate(self):
        return True

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, CausalLMOutput]:

        assert inputs_embeds is None, "inputs_embeds is not supported for now"
        assert past_key_values is None, "past_key_values is not supported for now"
        assert not use_cache, "use_cache is not supported for now"

        sequence_output, contextualized_embeddings = self.get_contextualized_embeddings(input_ids, attention_mask, output_hidden_states)
        subword_prediction = self.classifier(sequence_output)
        subword_prediction = 30 * torch.sigmoid(subword_prediction / 7.5)

        causal_lm_loss = None
        if labels is not None:
            labels_flatten = labels[:, 1:].flatten()
            subword_prediction_flatten = subword_prediction[:, :-1].flatten(0, 1)
            causal_lm_loss = F.cross_entropy(subword_prediction_flatten, labels_flatten)

        if not return_dict:
            output = (
                subword_prediction,
                *([contextualized_embeddings] if output_hidden_states else [])
            )
            return ((causal_lm_loss,) + output) if masked_lm_loss is not None else output

        return CausalLMOutput(
            loss=causal_lm_loss,
            logits=subword_prediction,
            hidden_states=contextualized_embeddings if output_hidden_states else None
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
        num_logits_to_keep: Optional[int] = None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


class GptBertForSequenceClassification(GptBertModel):
    _keys_to_ignore_on_load_missing = ["classifier.emb2vocab.weight", "classifier.emb2vocab.bias"]
    _keys_to_ignore_on_load_unexpected = ["classifier.emb2vocab.weight", "classifier.emb2vocab.bias"]

    def __init__(self, config: GptBertConfig, **kwargs):
        super().__init__(config, add_mlm_layer=False, **kwargs)

        self.num_labels = config.num_labels
        self.classifier = Classifier(config, self.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings = self.get_contextualized_embeddings(input_ids, attention_mask, output_hidden_states)
        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (
                logits,
                *([contextualized_embeddings] if output_hidden_states else [])
            )
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=contextualized_embeddings if output_hidden_states else None
        )


class GptBertForTokenClassification(GptBertModel):
    _keys_to_ignore_on_load_missing = ["classifier.emb2vocab.weight", "classifier.emb2vocab.bias"]
    _keys_to_ignore_on_load_unexpected = ["classifier.emb2vocab.weight", "classifier.emb2vocab.bias"]

    def __init__(self, config: GptBertConfig, **kwargs):
        super().__init__(config, add_mlm_layer=False, **kwargs)

        self.num_labels = config.num_labels
        self.classifier = Classifier(config, self.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings = self.get_contextualized_embeddings(input_ids, attention_mask, output_hidden_states)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (
                logits,
                *([contextualized_embeddings] if output_hidden_states else []),
                *([attention_probs] if output_attentions else [])
            )
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=contextualized_embeddings if output_hidden_states else None,
            attentions=attention_probs if output_attentions else None
        )


class GptBertForQuestionAnswering(GptBertModel):
    _keys_to_ignore_on_load_missing = ["classifier.emb2vocab.weight", "classifier.emb2vocab.bias"]
    _keys_to_ignore_on_load_unexpected = ["classifier.emb2vocab.weight", "classifier.emb2vocab.bias"]

    def __init__(self, config: GptBertConfig, **kwargs):
        super().__init__(config, add_mlm_layer=False, **kwargs)

        self.num_labels = config.num_labels
        self.classifier = Classifier(config, self.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        sequence_output, contextualized_embeddings = self.get_contextualized_embeddings(input_ids, attention_mask, output_hidden_states)
        logits = self.classifier(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (
                start_logits,
                end_logits,
                *([contextualized_embeddings] if output_hidden_states else [])
            )
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=contextualized_embeddings if output_hidden_states else None
        )


class GptBertForMultipleChoice(GptBertModel):
    _keys_to_ignore_on_load_missing = ["classifier.emb2vocab.weight", "classifier.emb2vocab.bias"]
    _keys_to_ignore_on_load_unexpected = ["classifier.emb2vocab.weight", "classifier.emb2vocab.bias"]

    def __init__(self, config: GptBertConfig, **kwargs):
        super().__init__(config, add_mlm_layer=False, **kwargs)

        self.num_labels = getattr(config, "num_labels", 2)
        self.classifier = Classifier(config, self.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        sequence_output, contextualized_embeddings = self.get_contextualized_embeddings(flat_input_ids, flat_attention_mask, output_hidden_states)
        logits = self.classifier(sequence_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (
                reshaped_logits,
                *([contextualized_embeddings] if output_hidden_states else [])
            )
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=contextualized_embeddings if output_hidden_states else None
        )
