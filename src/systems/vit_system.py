import pytorch_lightning as pl
import torch 
from torch import nn
from torch.nn import functional as F
from transformers import ViTForImageClassification, ViTFeatureExtractor
from omegaconf import DictConfig
import numpy as np


class ViTSystem(pl.LightningModule):
    
    def __init__(self, config: DictConfig):
        """
        Instantiates a trainable lightning module for cat/dog classification.
        Parameters:
            config: A config file matching the format in /catformer/scripts/config.yaml
        """
        super().__init__()
        
        self.config = config
        self.save_hyperparameters(config)

        pretrained_vit = ViTForImageClassification.from_pretrained(
            self.config.pretrained_model_name_or_path
        )
        feature_extractor = ViTFeatureExtractor.from_pretrained(
            self.config.pretrained_model_name_or_path
        )

        self.pretrained_vit_config = pretrained_vit.config
        self.vit = pretrained_vit.vit
        self.feature_extractor = feature_extractor

        if self.config.freeze_encoder_weights:
            for parameter in list(self.vit.parameters()):   
                parameter.requires_grad = False

        self.classifier = nn.Linear(
            self.pretrained_vit_config.hidden_size, self.config.num_classes
        )

        self.loss_fn = F.cross_entropy
        
    def forward(self, img):

        pixel_values = self.feature_extractor(img)['pixel_values']

        transformer_output = self.vit(
            pixel_values=pixel_values,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
        
        last_hidden_layer = transformer_output['last_hidden_state']
        attentions = transformer_output['attentions']
        hidden_states = transformer_output['hidden_states']

        # extract cls token
        pooled = last_hidden_layer[:, 0, :]

        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1).detach().numpy()

        predictions = (lambda idx: self.config.id2label[idx])(np.argmax(probs, dim=-1))

        return {
            'logits': logits,
            'probs': probs,
            'predictions': predictions,
            'attentions': attentions,
            'hidden_states': hidden_states
        }

    def training_step(self, batch, batch_idx):
        
        img, label = batch
        logits = self(img)['logits']
        y_hat = torch.argmax(logits, dim=-1)
        batch_size = y_hat.shape[0]
        
        loss = self.loss_fn(logits, y)
        accuracy = torch.sum(y_hat == y)/batch_size
        
        self.log('loss', loss, prog_bar=True, on_step=True)
        self.log('accuracy', accuracy, on_step=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        logits = self(x)[0]
        y_hat = torch.argmax(logits, dim=-1)
        batch_size = y_hat.shape[0]
        
        loss = self.loss_fn(logits, y)
        accuracy = torch.sum(y_hat == y)/batch_size
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_accuracy', accuracy, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    
    