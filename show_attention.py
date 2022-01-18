
from cmd import PROMPT
from xml.dom.expatbuilder import theDOMImplementation
import typer
import torch 
import einops
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms as T
from skimage.transform import resize
from skimage.exposure import adjust_gamma, equalize_hist

from main import DATA_DIRECTORY, DEFAULT_MODEL_PATH
from src.systems.vit_system import ViTSystem

CKPT_PATH = '/Users/paulwilson/lightning_logs/default/version_0/checkpoints/epoch=0-step=624.ckpt'

system = ViTSystem(
    DEFAULT_MODEL_PATH, 
    DATA_DIRECTORY, 
    batch_size=32,
)

checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'))
system.load_state_dict(checkpoint['state_dict'])

system.prepare_data()
system.setup(stage='predict')

def predict(system: ViTSystem, idx):
    
    dl = system.test_dataloader()
    
    X, y = dl.dataset[idx]
    
    # get unnormalized image
    temp = dl.dataset.transform
    temp_transform = T.Compose([
        T.Resize((system.config.image_size,
                  system.config.image_size)), 
        T.ToTensor(), 
        T.Lambda(
            lambda X: einops.rearrange(X, 'c h w -> h w c')
        ),
    ])
    dl.dataset.transform=temp_transform
    array, _ = dl.dataset[idx]
    array = array.numpy()
    dl.dataset.transform = temp

    batch = einops.repeat(X, 'c h w -> 1 c h w')
    logits, attentions = system(batch, output_attentions=True)
    
    probs = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()[0]
    pred = torch.argmax(logits, dim=-1)
    
    return {
        'array': array,
        'probs': probs, 
        'pred': pred,
        'attentions': attentions
    }
    
    
def get_class_attention_on_patches(attentions):
    
    b, h, n, _ = attentions[0].shape
    
    attentions = tuple(map(
        lambda matrix: (einops.reduce(matrix, 'b h n1 n2 -> n1 n2', 'mean') \
            + np.eye(n))/2, 
        attentions
    ))
    
    rollback = np.linalg.multi_dot(attentions)
    
    cls_attends_to = rollback[0, 1:]
    
    cls_attends_to_patches = einops.rearrange(
        cls_attends_to, '(p1 p2) -> p1 p2', p1=int(n**(1/2))
    )
    
    return cls_attends_to_patches
    
    
def plot_image_with_attention(idx, gamma=1, threshold=0):
    
    output = predict(system, idx)
    attn = get_class_attention_on_patches(output['attentions'])
    probs = output['probs']
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    ax1.set_title(
        f'Prediction: CAT {probs[0]*100:.2f}%, DOG {probs[1]*100:.2f}%'
    )
    ax1.imshow(output['array'])
    
    ax2.set_title('Attentions')
    ax2.imshow(attn)
    
    attn = equalize_hist(attn)
    attn = np.where(attn>threshold, attn, 0)
    
    attn_rescaled = resize(attn, (system.config.image_size,)*2, order=3)
    attn_rescaled = einops.repeat(attn_rescaled, 'h w -> h w 1')
    
    
    overlaid = equalize_hist(output['array']) * attn_rescaled
    overlaid = adjust_gamma(overlaid, gamma=gamma)
    ax3.set_title('overlaid')
    ax3.imshow(overlaid)
    
    plt.tight_layout
    plt.show()
    

def main(idx: int = typer.Option(..., prompt=True)):
    plot_image_with_attention(idx)
    
    typer.run(main)
    
if __name__ == '__main__':
    typer.run(main)