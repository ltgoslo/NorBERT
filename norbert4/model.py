from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from functools import partial

import math

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from config import ModelConfig


class ModelOutput:

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        loss: torch.Tensor | float | None = None,
        perplexity: torch.Tensor | float | None = None,
        accuracy: float | None = None,
        z_loss: torch.Tensor | float | None = None,
        **kwargs
    ):
        self.logits: torch.Tensor | None
        self.loss: torch.Tensor | float | None
        self.perplexity: torch.Tensor | float | None
        self.accuracy: float | None
        self.z_loss: torch.Tensor | float | None

        self.logits = logits
        self.loss = loss
        self.perplexity = perplexity
        self.accuracy = accuracy
        self.z_loss = z_loss

        for attr, value in kwargs.items():
            setattr(self, attr, value)


class CastedLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias=bias)

    def reset_parameters(self) -> None:
        std: float = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), bias=self.bias.type_as(x) if self.bias is not None else None)


class CastedLinearIn(nn.Linear):

    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias=bias)
        self.scale = nn.Parameter(torch.zeros(in_features))

    def reset_parameters(self) -> None:
        std: float = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return F.linear(x, (self.weight * (self.scale + 1.0).unsqueeze(0)).type_as(x), bias=self.bias.type_as(x) if self.bias is not None else None)


class CastedLinearOut(nn.Linear):

    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias=bias)
        self.scale = nn.Parameter(torch.ones(out_features))

    def reset_parameters(self) -> None:
        std: float = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return F.linear(x, (self.scale.unsqueeze(1) * self.weight).type_as(x), bias=self.bias.type_as(x) if self.bias is not None else None)


class MultiCastedLinearOrtho(nn.Module):

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

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i, weight in enumerate(self.weights):
            std: float = math.sqrt(2.0 / (self.in_features + self.out_features[i]))
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, x):
        return F.linear(x, torch.cat([weight for weight in self.weights], dim=0).type_as(x), bias=self.bias.type_as(x) if self.bias is not None else None)


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

        self.scale = nn.Parameter(torch.zeros(in_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weights:
            std = 0.5 * (self.in_features ** -0.5)
            bound = (3 ** 0.5) * std
            with torch.no_grad():
                weight.uniform_(-bound, bound)

    def forward(self, x):
        return F.linear(x, (torch.cat([weight for weight in self.weights], dim=0) * (self.scale + 1.0).unsqueeze(0)).type_as(x), bias=self.bias.type_as(x) if self.bias is not None else None)


class MultiCastedLinearOrthoOut(nn.Module):

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

        self.scale = nn.Parameter(torch.ones(sum(out_features)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.weights:
            std = 0.5 * (self.in_features ** -0.5)
            bound = (3 ** 0.5) * std
            with torch.no_grad():
                weight.uniform_(-bound, bound)

    def forward(self, x):
        return F.linear(x, (self.scale.unsqueeze(1) * torch.cat([weight for weight in self.weights], dim=0)).type_as(x), bias=self.bias.type_as(x) if self.bias is not None else None)


class GeGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        x = x * F.gelu(gate, approximate='tanh')
        return x


class Model(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.embedding: Embedding
        self.encoder: Encoder
        self.classifier: Classifier

        self.embedding = Embedding(config)
        self.encoder = Encoder(config)
        self.classifier = Classifier(config, self.embedding.word_embedding.weight)

        self.set_window_length(config.window_length)

    def set_window_length(self, window_length: int) -> None:
        self.encoder.set_window_length(window_length)

    def forward(self, input_ids: torch.Tensor, doc_ids: torch.Tensor, labels: torch.Tensor | None = None):
        word_embeddings: torch.Tensor
        encodings: torch.Tensor
        logits: torch.Tensor
        gold_labels: torch.Tensor
        output: ModelOutput

        word_embeddings = self.embedding(input_ids).bfloat16()
        encodings = self.encoder(word_embeddings, word_embeddings, doc_ids).bfloat16()
        logits = self.classifier(encodings, labels).float()
        logits = 30 * torch.sigmoid(logits / 7.5)

        output = ModelOutput(logits=logits, loss=None, perplexity=None, z_loss=None, accuracy=None, num_tokens=None)

        if labels is not None:

            gold_labels = labels.flatten()
            gold_labels = gold_labels[gold_labels != -100]

            output.loss = F.cross_entropy(logits, gold_labels)
            output.perplexity = torch.exp(output.loss)
            output.z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()

            with torch.no_grad():
                output.accuracy = (logits.argmax(-1) == gold_labels).float().mean()

            output.num_tokens = gold_labels.size(0)

        return output


class Encoder(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.layers: nn.ModuleList[Layer]
        self.short_long_ratio = config.short_long_ratio

        self.layers = nn.ModuleList([Layer(config, i) for i in range(config.num_layers)])

        for i, layer in enumerate(self.layers):
            for weight in layer.mlp.up_proj.weights:
                weight.data *= math.sqrt(1.0 / (2.0 * (i + 1)))
            layer.mlp.down_proj.weight.data *= math.sqrt(1.0 / (2.0 * (i + 1)))

    def set_window_length(self, window_length: int) -> None:
        for i, layer in enumerate(self.layers):
            if (i+1) % self.short_long_ratio == 0:
                layer.set_window_length(window_length)
            else:
                layer.set_window_length(256)

    def forward(self, hidden_layer: torch.Tensor, embeddings: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        v1 = None

        for i, layer in enumerate(self.layers):
            hidden_layer, v1 = layer(hidden_layer, embeddings, v1, doc_ids)

        return hidden_layer


class Layer(nn.Module):

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()

        self.attention: SelfAttention
        self.mlp: FeedForward

        self.attention = SelfAttention(config, layer_idx)
        self.mlp = FeedForward(config)
        self.lambdas_v = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.lambdas_qk = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.lambdas_mlp = nn.Parameter(torch.tensor([1.0, 1.0, 0.0]))
        self.lambdas_out = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 0.0]))

    def set_window_length(self, window_length: int) -> None:
        self.attention.set_window_length(window_length)

    def normalize_lambda(self, lambdas: torch.Tensor) -> torch.Tensor:
        lambdas = lambdas / (lambdas.abs().mean() + 1e-6)
        return lambdas

    def forward(self, hidden_layer: torch.Tensor, embeddings: torch.Tensor, v1: torch.Tensor | None, doc_ids: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor

        lambdas_v = self.normalize_lambda(self.lambdas_v)
        lambdas_qk = self.normalize_lambda(self.lambdas_qk)
        lambdas_mlp = self.normalize_lambda(self.lambdas_mlp)
        lambdas_out = self.normalize_lambda(self.lambdas_out)

        v_layer = (lambdas_v[0] * hidden_layer) + (lambdas_v[1] * embeddings)
        qk_layer = (lambdas_qk[0] * hidden_layer) + (lambdas_qk[1] * embeddings)
        attention_output, v1 = self.attention(v_layer, qk_layer, v1, doc_ids)

        mlp_layer = (lambdas_mlp[0] * attention_output) + (lambdas_mlp[1] * hidden_layer) + (lambdas_mlp[2] * embeddings)
        mlp_layer = self.mlp(mlp_layer)

        output = (lambdas_out[0] * mlp_layer) + (lambdas_out[1] * attention_output) + (lambdas_out[2] * hidden_layer) + (lambdas_out[3] * embeddings)

        return output, v1


class Embedding(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        assert hasattr(config, "vocab_size"), "The config must have a vocab_size attribute!"
        assert hasattr(config, "hidden_size"), "The config must have a hidden_size attribute!"
        assert hasattr(config, "embedding_dropout_p"), "The model must have a embedding_dropout_p attribute!"

        self.word_embedding: nn.Embedding
        self.word_norm: nn.LayerNorm
        self.dropout: nn.Dropout

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.word_norm = nn.LayerNorm(config.hidden_size, eps=config.word_norm_eps, elementwise_affine=False, bias=False)
        self.word_scale = nn.Parameter(torch.zeros(config.hidden_size))

        # self.dropout = nn.Dropout(config.embedding_dropout_p)

        self.initialize(config.hidden_size, config.vocab_size)

    @torch.no_grad()
    def initialize(self, hidden_size: int, vocab_size: int) -> None:
        std: float

        std = math.sqrt(2.0 / (hidden_size + vocab_size))
        nn.init.trunc_normal_(self.word_embedding.weight, mean=0.0, std=std, a=-2*std, b=2*std)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        word_embedding: torch.Tensor

        word_embedding = self.word_embedding(input_ids)
        word_embedding = self.word_norm(word_embedding)
        word_embedding = (word_embedding * (self.word_scale + 1.0).unsqueeze(0).unsqueeze(0))

        return word_embedding


class Classifier(nn.Module):

    def __init__(self, config: ModelConfig, embedding_weights: nn.Parameter) -> None:
        super().__init__()

        self.projection: CastedLinear
        self.emb2vocab: CastedLinear
        self.pre_norm: nn.LayerNorm
        self.post_norm: nn.LayerNorm

        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=config.classifier_pre_norm_eps, elementwise_affine=config.classifier_pre_norm_affine)
        self.projection = CastedLinearIn(config.hidden_size, config.hidden_size, bias=False)
        self.post_norm = nn.LayerNorm(config.hidden_size, eps=config.classifier_post_norm_eps, elementwise_affine=config.classifier_post_norm_affine)
        self.emb2vocab = CastedLinearIn(config.hidden_size, config.vocab_size, bias=True)

        self.initialize(config.hidden_size, config.vocab_size, embedding_weights)

    @torch.no_grad()
    def initialize(self, hidden_size: int, vocab_size: int, embedding_weights: nn.Parameter) -> None:
        proj_std: float = math.sqrt(2.0 / (hidden_size + 4*hidden_size))

        nn.init.trunc_normal_(self.projection.weight, mean=0.0, std=proj_std, a=-2*proj_std, b=2*proj_std)
        self.emb2vocab.weight = embedding_weights
        self.emb2vocab.bias.zero_()

    def project(self, hidden_layer: torch.Tensor) -> torch.Tensor:
        projection: torch.Tensor

        projection = self.projection(hidden_layer)
        projection = F.gelu(projection, approximate='tanh')
        projection = self.post_norm(projection.float()).type_as(hidden_layer)

        return projection

    def calculate_output(self, hidden_layer: torch.Tensor) -> torch.Tensor:
        return self.emb2vocab(hidden_layer)

    def forward(self, hidden_layer: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        output: torch.Tensor

        if labels is not None:
            hidden_layer = torch.index_select(hidden_layer.flatten(0, 1), 0, torch.nonzero(labels.flatten() != -100).squeeze())

        hidden_layer = self.pre_norm(hidden_layer.float()).type_as(hidden_layer)
        hidden_layer = self.project(hidden_layer)
        output = self.calculate_output(hidden_layer)

        return output


class SelfAttention(nn.Module):

    def __init__(self, config: ModelConfig, layer_idx) -> None:
        super().__init__()
        self.d_qk = config.d_qk
        self.d_v = config.d_v
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.hidden_size = config.hidden_size

        self.q_out_dim = self.d_qk * self.num_attention_heads
        self.k_out_dim = self.d_qk * self.num_kv_heads
        self.v_out_dim = self.d_v * self.num_kv_heads

        self.qk_proj = MultiCastedLinearOrthoIn(self.hidden_size, [self.q_out_dim, self.k_out_dim], bias=False)
        self.v_proj = CastedLinearIn(self.hidden_size, self.v_out_dim, bias=False)
        self.out_proj = CastedLinearIn(self.d_v*self.num_attention_heads, self.hidden_size, bias=False)

        self.pre_v_norm = nn.LayerNorm(config.hidden_size, eps=config.attention_pre_norm_eps, elementwise_affine=config.attention_pre_norm_affine)
        self.pre_qk_norm = nn.LayerNorm(config.hidden_size, eps=config.attention_pre_norm_eps, elementwise_affine=config.attention_pre_norm_affine)
        self.inter_norm = nn.LayerNorm(self.d_v * self.num_attention_heads, eps=config.attention_inter_norm_eps, elementwise_affine=config.attention_inter_norm_affine, bias=False)
        self.q_norm = nn.LayerNorm(config.d_qk, eps=config.attention_pre_norm_eps, elementwise_affine=False, bias=False)
        self.k_norm = nn.LayerNorm(config.d_qk, eps=config.attention_pre_norm_eps, elementwise_affine=False, bias=False)
        self.k_scale = nn.Parameter(torch.zeros(self.num_kv_heads, config.d_qk))
        self.q_scale = nn.Parameter(torch.zeros(self.num_attention_heads, config.d_qk))

        self.dropout = nn.Dropout(config.attention_output_dropout_p)

        theta = 160_000 if (layer_idx + 1) % config.short_long_ratio == 0 else 10_000

        self.rope_embedding = RotaryPositionalEmbeddings(config, theta)
        self.scale: float = 1.0 / math.sqrt(self.d_qk)

        # self.lambdas = nn.Parameter(torch.tensor([0.5]))

        self.initialize()

        self.sequence_length = config.max_sequence_length
        self.is_causal = config.dataset_type == "causal"

    @torch.no_grad()
    def initialize(self) -> None:
        std: float = math.sqrt(2.0 / (self.hidden_size + 4*self.hidden_size))
        for weight in self.qk_proj.weights:
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2*std, b=2*std)
        nn.init.trunc_normal_(self.v_proj.weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.out_proj.weight.data.zero_()

    def set_window_length(self, window_length: int) -> None:
        self.window_length: int = window_length
        self.block_mask = self.create_block_mask(window_length)

    def causal_mask_mode(self, window_length, b, _, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) < window_length)

    def bidirectional_mask_mode(self, window_length, b, _, q_idx, kv_idx):
        return ((q_idx - kv_idx) < window_length) & ((kv_idx - q_idx) < window_length)

    def create_block_mask(self, window_length: int) -> torch.Tensor:
        if self.is_causal:
            return create_block_mask(
                partial(self.causal_mask_mode, self.window_length),
                1, 1, self.sequence_length, self.sequence_length
            )
        else:
            return create_block_mask(
                partial(self.bidirectional_mask_mode, self.window_length),
                1, 1, self.sequence_length, self.sequence_length
            )

    def forward(self, hidden_layer: torch.Tensor, qk_layer: torch.Tensor, v1: torch.Tensor | None, doc_ids: torch.Tensor) -> torch.Tensor:
        hidden_layer = self.pre_v_norm(hidden_layer.float()).type_as(hidden_layer)
        qk_layer = self.pre_qk_norm(qk_layer.float()).type_as(qk_layer)

        query, key = self.qk_proj(qk_layer).tensor_split([self.q_out_dim], dim=-1)  # shape: [T, B, H*D]
        value = self.v_proj(hidden_layer)

        query_length: int = hidden_layer.size(0)
        key_length: int = hidden_layer.size(0)
        batch_size: int = hidden_layer.size(1)

        query = query.reshape(query_length, batch_size, self.num_attention_heads, self.d_qk).permute(1, 2, 0, 3)  # shape: [B, H, T, D]
        key = key.reshape(key_length, batch_size, self.num_kv_heads, self.d_qk).permute(1, 2, 0, 3)  # shape: [B, H, T, D]
        value = value.reshape(key_length, batch_size, self.num_kv_heads, self.d_v).permute(1, 2, 0, 3)  # shape: [B, H, T, D]

        query, key = ((self.q_scale + 1.0).unsqueeze(1).unsqueeze(0) * self.q_norm(query.float())).type_as(query), ((self.k_scale + 1.0).unsqueeze(1).unsqueeze(0) * self.k_norm(key.float())).type_as(key)

        # if v1 is None:
        #     v1 = value
        # value = (1 - self.lambdas[0]) * value + self.lambdas[0] * v1

        query = self.rope_embedding(query)
        key = self.rope_embedding(key)

        def document_score_mod(score, b, _, q_idx, kv_idx):
            return torch.where(doc_ids[q_idx] == doc_ids[kv_idx], score, -float("inf"))

        output = flex_attention(
            query, key, value, block_mask=self.block_mask, enable_gqa=True, score_mod=document_score_mod
        )

        output = output.permute(2, 0, 1, 3).flatten(2, 3)  # shape: [T, B, H*D]
        output = self.inter_norm(output.float()).type_as(hidden_layer)
        output = self.out_proj(output)

        return self.dropout(output), v1


class FeedForward(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.up_proj: CastedLinear
        self.down_proj: CastedLinear
        self.pre_norm: nn.LayerNorm
        self.inter_norm: nn.LayerNorm
        self.activation: GeGLU
        self.dropout: nn.Dropout

        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=config.feed_forward_pre_norm_eps, elementwise_affine=config.feed_forward_pre_norm_affine)
        self.up_proj = MultiCastedLinearOrthoIn(config.hidden_size, [config.intermediate_size, config.intermediate_size], bias=False)
        self.activation = GeGLU()
        self.inter_norm = nn.LayerNorm(config.intermediate_size, eps=config.feed_forward_inter_norm_eps, elementwise_affine=config.feed_forward_inter_norm_affine)
        self.down_proj = CastedLinearIn(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.feed_forward_dropout_p)

        self.initialize(config.hidden_size)

    @torch.no_grad()
    def initialize(self, hidden_size: int) -> None:
        std: float = math.sqrt(2.0 / (5*hidden_size))

        for weight in self.up_proj.weights:
            nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2*std, b=2*std)
        self.down_proj.weight.data.zero_()

    def up_project(self, hidden_layer: torch.Tensor) -> torch.Tensor:
        hidden_layer = self.pre_norm(hidden_layer.float()).type_as(hidden_layer)
        return self.up_proj(hidden_layer)

    def activate(self, projection: torch.Tensor) -> torch.Tensor:
        activated_projection: torch.Tensor

        activated_projection = self.activation(projection)
        activated_projection = self.inter_norm(activated_projection.float()).type_as(projection)

        return activated_projection

    def down_project(self, activated_projection: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor

        output = self.down_proj(activated_projection)

        return self.dropout(output)

    def forward(self, hidden_layer: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor

        output = self.up_project(hidden_layer)
        output = self.activate(output)
        output = self.down_project(output)

        return output


class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, config: ModelConfig, theta: int) -> None:
        super().__init__()

        assert hasattr(config, "d_qk"), "The config must have a d_qk attribute!"
        assert hasattr(config, "max_sequence_length"), "The config must have a max_sequence_length attribute!"

        self.inv_freq: torch.Tensor
        self.cos_matrix: torch.Tensor
        self.sin_matrix: torch.Tensor
        head_size: int
        max_seq_len: int
        inv_freq: torch.Tensor
        pos: torch.Tensor
        embedding: torch.Tensor

        head_size = config.d_qk
        assert head_size % 2 == 0
        max_seq_len = config.max_sequence_length

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_size, 2, dtype=torch.float32) / head_size))
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        embedding = torch.einsum('n, d -> nd', pos, inv_freq)
        embedding = torch.cat([embedding, embedding], dim=-1).unsqueeze(0)
        self.register_buffer("cos_matrix", embedding.cos(), persistent=False)
        self.register_buffer("sin_matrix", embedding.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len: int
        cos_matrix: torch.Tensor
        sin_matrix: torch.Tensor
        x_rotate_half: torch.Tensor
        out: torch.Tensor

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

