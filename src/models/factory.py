
from omegaconf import DictConfig

class ModelFactory:
    def __init__(self, config: DictConfig):
        self.config = config

    def build_model(self):
        if self.config.name == 'vit':
            from src.models.vit import ViT

            model = ViT(
                self.config.input_shape,
                self.config.num_channels,
                self.config.patch_size,
                self.config.num_classes,
                self.config.embedding_dim,
                self.config.num_heads,
                self.config.depth,
                self.config.head_dim,
                self.config.mlp_dim,
                self.config.dropout
            )
        else:
            raise NotImplementedError(f'model {self.config.name} is not supported.')

        return model

