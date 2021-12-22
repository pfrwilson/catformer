import einops
import torch
from torch.nn.functional import interpolate
import torchvision.transforms
from torch import nn
import einops
from einops.layers.torch import Rearrange
from src.models.transformer import Transformer


class ViTBase(nn.Module):
    def __init__(self, input_shape: tuple[int, int], num_channels: int,
                 embedding_dim: int,
                 to_embeddings: nn.Module,
                 attention_module: nn.Module,
                 num_classes: int,
                 use_pos_embedding: bool = True):
        super(ViTBase, self).__init__()

        self.input_shape = input_shape
        self.num_channels = num_channels
        self.to_embeddings = to_embeddings
        self.attention_module = attention_module
        self.pos_embedding = nn.Parameter(torch.zeros((self.num_patches + 1, embedding_dim)), requires_grad=False)
        if use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.randn((self.num_patches + 1, embedding_dim)))

        self.fc = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self, x, cache_attn=False):
        b, _, _, _ = x.shape
        x = self.to_embeddings(x)
        x = torch.concat(
            [torch.zeros((b, 1, x.shape[-1])), x],
            dim=-2
        )  # add space for class embedding
        x = x + self.pos_embedding
        x = self.attention_module(x, cache_attn=cache_attn)
        x = x[:, 0, :]  # class token
        logits = self.fc(x)
        return logits

    def get_attn_rollback(self, x, upsample=True):
        with torch.no_grad():
            self(x, cache_attn=True)
        attn_maps = self.attention_module.cached_attn
        attn = torch.eye(self.num_patches + 1)
        for attn_map in attn_maps:
            attn = torch.matmul(attn, attn_map)

        attn = attn[:, 0, 1:]  # attn to class token

        attn = einops.rearrange(attn, 'b (p_h p_w) -> b p_h p_w',
                                p_h=self.input_shape[0]//self.patch_size[0],
                                p_w=self.input_shape[1]//self.patch_size[1])
        attn = einops.repeat(attn, 'b p_h p_w -> b c p_h p_w', c=self.num_channels)

        if upsample:
            attn = interpolate(attn, self.input_shape)

        return attn
