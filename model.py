from torch import nn
import torch
from x_transformers import AttentionLayers

TRUNC_STD = 0.02
class Regularizer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.normalize(x, p=2, dim=-1)

# some more parameters to add
# selective = False
# residual_attn = False
# use_dynamic_tanh = False only for prenorm 

class PatchProjection(nn.Module):
    def __init__(self, inp_shape,d_model):
        super().__init__()
        self.patch_proj = nn.Linear(in_features=inp_shape, out_features=d_model)
        # nn.init.trunc_normal_(self.patch_proj.weight, std=TRUNC_STD)
    def forward(self, x):
        return self.patch_proj(x)


class KWSTransformer(nn.Module):
    def __init__(self,
        num_patches,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        channels=3,
        dropout=0.1,
        prenorm=False,
        distill_token=False,
        approximate_gelu=False,
        rotary = False,
        res_attn = False,
        selective_attn = False,
        use_dynamic_tanh = False
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.distill_token = distill_token
        self.rotary = rotary
        additional_tokens = 2 if distill_token else 1
        self.pos_emb = nn.Parameter(torch.empty(1, num_patches + additional_tokens, d_model))
        self.class_emb = nn.Parameter(torch.empty(1, 1, d_model))
        self.distill_emb = nn.Parameter(torch.empty(1, 1, d_model)) if distill_token else None
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.class_emb, std=0.02)
        self.enc_layers = AttentionLayers(causal = False,dim = d_model, heads = num_heads, pre_norm = prenorm, depth = num_layers, rotary_pos_emb = rotary, rotary_emb_dim = d_model, residual_attn = res_attn, use_dynamic_tanh = use_dynamic_tanh, dynamic_tanh_init_alpha = 1, selective_attn = selective_attn)

    def forward(self, x):
        batch_size = x.shape[0]

        class_emb = self.class_emb.expand(batch_size, 1, self.d_model)
        if self.distill_token:
            distill_emb = self.distill_emb.expand(batch_size, 1, self.d_model)
            tokens = [class_emb, distill_emb, x]
        else:
            tokens = [class_emb, x]
        # print("concated",tokens)
        x = torch.cat(tokens, dim=1)
        # print("latest")
        if not self.rotary:
            x = x + self.pos_emb

        x = self.enc_layers(x)
        # print("aft attention", x)
        class_output = x[:, 0]
        if self.distill_token:
            distill_output = x[:, 1]
            return class_output, distill_output
        else:
            return class_output
