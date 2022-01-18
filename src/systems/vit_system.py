import pytorch_lightning as pl
import torch 
from torch import nn 
from torchmetrics import Accuracy
from transformers import ViTForImageClassification
from omegaconf import DictConfig
from ..data.cats_dogs import DogsVsCats
from torch.utils.data import random_split, DataLoader
from ..data.utils import TransformDataset
from omegaconf import OmegaConf


class ViTSystem(pl.LightningModule):
    
    def __init__(self, config: OmegaConf):
        """
        Instantiates a trainable lightning module for cat/dog classification.
        Parameters:
            config: A config file matching the format in /catformer/scripts/config.yaml
        """
        super().__init__()
        
        self.config = config
        self.save_hyperparameters(config)
        
        # pretrained vit model
        self.pretrained_vit = ViTForImageClassification.from_pretrained(
            self.config.pretrained_model_name_or_path
        )
        
        if self.config.freeze_encoder_weights:
            for parameter in list(self.pretrained_vit.vit.parameters()):   
                parameter.requires_grad = False
    
        self.id2label = {
            0: 'cat', 
            1: 'dog'
        }
        self.label2id = {
            'cat': 0, 
            'dog': 1
        }
        self.classifier = nn.Linear(
            self.pretrained_vit.config.hidden_size, 2
        )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, x, output_attentions=False):
    
        transformer_output = self.pretrained_vit.vit(
            x, output_attentions=output_attentions
        )
        
        last_hidden_layer = transformer_output[0]
        attentions = transformer_output[1] if output_attentions else None
        
        class_token = last_hidden_layer[:, 0, :]

        logits = self.classifier(class_token)
        
        return logits, attentions
        
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        logits = self(x)[0]
        y_hat = torch.argmax(logits, dim=-1)
        batch_size = y_hat.shape[0]
        
        loss = self.loss_fn(logits, y)
        accuracy = torch.sum(y_hat == y)/batch_size
        
        self.log('loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('accuracy', accuracy, on_epoch=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        logits = self(x)
        y_hat = torch.argmax(logits, dim=-1)
        batch_size = y_hat.shape[0]
        
        loss = self.loss_fn(logits, y)
        accuracy = torch.sum(y_hat == y)/batch_size
        
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_accuracy', accuracy, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    