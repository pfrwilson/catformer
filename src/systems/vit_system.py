import pytorch_lightning as pl
import torch 
from torch import nn 
from torchmetrics import Accuracy
from transformers.models.vit.modeling_vit import ViTForImageClassification

class ViTSystem(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        
    def forward(self):
        

