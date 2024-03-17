import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


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

class LinearAttention(nn.Module):
    def __init__(
        self,
        d_model: str = 1024,
        expand_k: int = 1.0,
        expand_v: int = 1.0,
        num_heads: int = 8,
        mode: str = 'chunk',
        feature_map: str = 'elementwise_product',
        tie_feature_map_qk: bool = False,
        output_norm: str = 'rmsnorm',
        norm_q: bool = False,
        norm_k: bool = False,
        # standard linear attention normalization
        do_feature_map_norm: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert feature_map in ['elu', 'relu', 'hedgehog', 't2r', 'dpfp',
                               'identity', 'elementwise_product'], f"Not supported feature map `{feature_map}`."

        assert output_norm in ['rmsnorm', 'identity'], f"Not supported output norm `{output_norm}`."

        self.d_model = d_model
        self.mode = mode
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.num_heads = num_heads

        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        if feature_map == 'hedgehog':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HedgehogFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 't2r':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = T2RFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'elementwise_product':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HadamardFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'dpfp':
            self.feature_map_q = DPFPFeatureMap(head_dim=self.head_qk_dim)
            self.feature_map_k = DPFPFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'elu':
            def elu(x):
                return F.elu(x) + 1
            self.feature_map_q = elu
            self.feature_map_k = elu

        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()

        elif feature_map == 'identity':
            self.feature_map_q = nn.Identity()
            self.feature_map_k = nn.Identity()
        else:
            raise NotImplementedError

        self.do_feature_map_norm = do_feature_map_norm
        if output_norm == 'rmsnorm':
            self.norm = RMSNorm(self.head_v_dim)
        elif output_norm == 'identity':
            self.norm = nn.Identity()
        else:
            raise NotImplementedError

        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)

        self.norm_q = norm_q
        self.norm_k = norm_k

    def forward(self, x):
        mode = self.mode
        q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        q = self.feature_map_q(q)
        k = self.feature_map_k(k)
        if self.norm_q:
            q = q / (q.sum(-1, keepdim=True) + 1e-4)
        if self.norm_k:
            k = k / (k.sum(-1, keepdim=True) + 1e-4)

        o = fused_chunk_linear_attn(q, k, v, normalize=self.do_feature_map_norm)
        o = self.norm(o)
        o = rearrange(o, 'b h n d -> b n (h d)')
        o = self.o_proj(o)
        return o
