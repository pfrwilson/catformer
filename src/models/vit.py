import einops
import torch
from torch.nn.functional import interpolate
import torchvision.transforms
from torch import nn
import einops
from einops.layers.torch import Rearrange
from src.models.transformer import Transformer
from src.models.vit_base import ViTBase


class ViT(ViTBase):
    def __init__(self, input_shape, num_channels, patch_size, num_classes,
                 embedding_dim, num_heads, depth, head_dim, mlp_dim, dropout=0.):

        self.patch_size = patch_size
        assert input_shape[0] % patch_size[0] == 0, 'patch sizes must divide input size'
        assert input_shape[1] % patch_size[1] == 0, 'patch sizes must divide input size'
        self.num_patches = (input_shape[0] // patch_size[0]) * (input_shape[1] // patch_size[1])

        to_patches: nn.Module = Rearrange(
            'b c (h p_h) (w p_w) -> b (h w) c p_h p_w',
            p_h=patch_size[0],
            p_w=patch_size[1]
        )

        to_patch_embeddings: nn.Module = nn.Sequential(
            Rearrange('b n_p c p_h p_w -> b n_p (c p_h p_w)', c=num_channels, p_h=patch_size[0],
                      p_w=patch_size[1]),
            nn.Linear(in_features=num_channels * patch_size[0] * patch_size[1], out_features=embedding_dim)
        )

        to_embeddings: nn.Module = nn.Sequential(
            to_patches,
            to_patch_embeddings
        )

        transformer = Transformer(embedding_dim, depth, num_heads, head_dim, mlp_dim, dropout=dropout)

        super(ViT, self).__init__(
            input_shape=input_shape,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            to_embeddings=to_embeddings,
            attention_module=transformer,
            num_classes=num_classes,
            use_pos_embedding=True
        )



