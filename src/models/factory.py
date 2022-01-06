import torch
from omegaconf import DictConfig


def build_model(name: str, config: dict):
    if name == 'vit':
        from src.models.vit import ViT

        try:
            model = ViT(
                config['input_shape'],
                config['num_channels'],
                config['patch_size'],
                config['num_classes'],
                config['embedding_dim'],
                config['num_heads'],
                config['depth'],
                config['head_dim'],
                config['mlp_dim'],
                config['dropout'],
            )
        except KeyError as e:
            raise KeyError(f'parameter {e.args[0]} not specified in config.')

    else:
        raise NotImplementedError(f'model {self.config.name} is not supported.')

    return model

