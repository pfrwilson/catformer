
from debugpy import trace_this_thread
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
import numpy as np
import einops
from PIL import Image
import matplotlib.pyplot as plt

from src.systems.vit_system import ViTSystem
from src.data.cats_dogs import DogsVsCats
from src.utils.attn_visualization import AttentionGrabber

DEVICE = torch.device('cuda' if torch.cude.is_available() else 'cpu')

@hydra.main(config_path='conf', config_name='predict')
def main(config: DictConfig):

    system = ViTSystem.load_from_checkpoint(config.model.checkpoint, map_location=DEVICE)
    
    to_model_input = DogsVsCats.get_default_transform(
                target_size=system.config.image_size,
                use_augmentations=False
            )
    
    img: Image
    pixel_values: torch.Tensor
        
    if config.data.mode == "from_dataset": 
        
        ds = DogsVsCats(
            root=config.data.data_root, 
            transform=to_model_input, 
            split=config.data.split
        )
        
        with ds.raw():
            img = ds[config.data.idx]
        
        pixel_values = ds[config.data.idx]
        
    elif config.data.mode == 'from_png':
        
        with open(config.data.img_root, 'rb') as f:
                img = Image.open(f)
                img.load()
        
        pixel_values = to_model_input(img)

    batch = einops.repeat(pixel_values, 'c h w -> 1 c h w')
    
    with torch.no_grad():
        output_dict = system(batch)
    
    logits = output_dict['logits']
    probs = F.softmax(logits, dim=-1)
    probs = probs.detach().cpu().numpy()
    probs = probs[0] # get rid of batch dimension
    prob, predicted = np.max(probs), np.argmax(probs)
    predicted = system.config.id2label[predicted]
    
    attentions = output_dict['attentions'].detach().cpu().numpy()
    map_grabber = AttentionGrabber(attentions)
    
    # VISUALIZATION  
    
    fig, ax1, ax2 = plt.subplots(1, 2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f'{predicted}, {prob:.2f}')
    ax1.imshow(np.array(img))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('attn map')
    ax2.imshow(map_grabber.get_attention_maps()[0])
    
    plt.tight_layout()
    plt.show()
    