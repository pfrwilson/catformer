import torch
from torch import nn

_MODELS = {}


def register_model(model: nn.Module, name: str):
    _MODELS[name] = model


def get_model(name: str):
    try:
        return _MODELS[name]
    except KeyError:
        print(f'model {name} not registered.')


def setup_model_store():

    # =======================
    from torchvision.models import resnet18
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(in_features=512, out_features=2)

    register_model(model, 'resnet_18')

    # ========================
    from . import vit
    params = {
        'input_shape': [256, 256],
        'num_channels': 3,
        'patch_size': [16, 16],
        'num_classes': 2,
        'embedding_dim': 128,
        'num_heads': 4,
        'depth': 4,
        'head_dim': 64,
        'mlp_dim': 64,
        'dropout': 0.2,
    }
    model = vit.ViT(**params)
    register_model(model, 'vit')

    # =======================


setup_model_store()

