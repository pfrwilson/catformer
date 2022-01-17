import pytorch_lightning as pl
import torch 
from torch import nn 
from torchmetrics import Accuracy
from transformers import ViTForImageClassification, ViTConfig
from omegaconf import DictConfig
from ..data.cats_dogs import DogsVsCats
from torch.utils.data import random_split, DataLoader
from ..data.utils import TransformDataset

class ViTSystem(pl.LightningModule):
    
    def __init__(self,
                 pretrained_model_name_or_path: str, 
                 data_root,
                 batch_size):
        """
        Instantiates a trainable lightning module for cat/dog classification.
        Parameters:
            config: A config file matching the format in /catformer/scripts/config.yaml
        """
        super().__init__()
        
        vit = ViTForImageClassification.from_pretrained(
            pretrained_model_name_or_path
        )
        self.data_root = data_root
        self.vit = vit.vit
        for parameter in list(self.vit.parameters()):
            parameter.requires_grad = False
        self.image_size = vit.config.image_size
        self.hidden_size = vit.config.hidden_size 
        self.classifier = nn.Linear(self.hidden_size, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.batch_size = batch_size
        
    def forward(self, x, output_attentions=False):
    
        transformer_output = self.vit(x)[0]
        class_token = transformer_output[:, 0, :]

        logits = self.classifier(class_token)
        
        return logits
        
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        logits = self(x)
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
        
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_accuracy', accuracy, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters())
    
    def setup(self, stage): 
        
        self.full_ds = DogsVsCats(self.data_root)
        
        train_length = int(0.8 * len(self.full_ds))
        test_length = len(self.full_ds) - train_length
        
        self.train_ds, self.val_ds = random_split(
            self.full_ds,
            [train_length, test_length], 
            generator = torch.Generator().manual_seed(42)
        )
        
        # Apply augmentations by wrapping in map dataset
        self.train_ds = TransformDataset(
            self.train_ds, 
            DogsVsCats.get_default_transform(
                target_size = self.image_size,
                use_augmentations = True
            )
        )
        self.val_ds = TransformDataset(
            self.val_ds, 
            DogsVsCats.get_default_transform(
                target_size = self.image_size, 
                use_augmentations=False
            )
        )
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)