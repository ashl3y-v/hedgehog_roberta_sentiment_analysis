import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Optional, Tuple
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention


class HedgehogFeatureMap(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()

        self.layer = nn.Linear(head_dim, head_dim)
        self.init_weights_()

    def init_weights_(self):
        """Initialize trainable map as identity"""
        nn.init.eye_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self, x: torch.Tensor):
        return nn.functional.softmax(self.layer(x), dim=-1)


class RobertaHedgehogSelfAttention(nn.Module):
    def __init__(self, base_attn: RobertaSelfAttention, training=False):
        super().__init__()

        self.base_attn = base_attn

        # for p in self.base_attn.parameters():
        #     p.requires_grad = False

        self.mlp_q = HedgehogFeatureMap(self.base_attn.all_head_size)
        self.mlp_k = HedgehogFeatureMap(self.base_attn.all_head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        **base_kwargs: Any
    ) -> Tuple[torch.Tensor]:
        k = self.base_attn.transpose_for_scores(
            self.mlp_k(self.base_attn.key(hidden_states))
        )
        v = self.base_attn.transpose_for_scores(self.base_attn.value(hidden_states))
        q = self.base_attn.transpose_for_scores(
            self.mlp_q(self.base_attn.query(hidden_states))
        )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_probs = torch.matmul(q, k.transpose(-1, -2))

        # Normalize the attention scores to probabilities.
        attention_probs /= attention_probs.sum(dim=-1, keepdim=True)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, v)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.base_attn.all_head_size,
        )
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs
