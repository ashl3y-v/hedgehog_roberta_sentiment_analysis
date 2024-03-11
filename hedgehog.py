import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from fla.ops.linear_attn import fused_chunk_linear_attn


def quadratic_linear_attention(q: torch.Tensor, k: torch.Tensor):
    qk = torch.einsum("bhmd,bhnd->bhmn", q, k)
    return qk / qk.sum(dim=-1, keepdim=True)


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
        x = self.layer(x)  # [b,h,l,d]
        return torch.cat([torch.exp(x), torch.exp(-x)], dim=-1)


class HedgehogAttention(nn.Module):
    def __init__(self, base_attn, training=True):
        self.base_attn = base_attn

        self.mlp_q = HedgehogFeatureMap(base_attn.head_dim)
        self.mlp_k = HedgehogFeatureMap(base_attn.head_dim)

        for p in self.base_attn.parameters():
            p.requires_grad = False

        self.q_proj = self.base_attn.q_proj
        self.k_proj = self.base_attn.k_proj
        self.v_proj = self.base_attn.v_proj

        self.training = training

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = True,
        **kwargs: any
    ):
        if self.training:
            outputs, true_attns = self.base_attn(
                hidden_states=hidden_states, output_attentions=True, **kwargs
            )

        q = self.mlp_q(self.q_proj(hidden_states))
        k = self.mlp_k(self.k_proj(hidden_states))
        v = self.v_proj(hidden_states)

        pred_attns = quadratic_linear_attention(q, k)

        if output_attentions:
            return outputs, (pred_attns, true_attns)
