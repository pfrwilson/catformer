import typer
from src.systems.vit_system import ViTSystem
import pytorch_lightning as pl
import os
from typing import Optional
import torch

DATA_DIRECTORY = os.path.join(
    os.environ['HOME'], 
    'data', 
    'dogs-vs-cats', 
    'train'
)

DEFAULT_MODEL_PATH = 'google/vit-large-patch16-384'

def main(
        data_directory: str = DATA_DIRECTORY, 
        num_gpus: Optional[int] = None, 
        max_epochs: Optional[int] = None, 
        model_path: str = DEFAULT_MODEL_PATH,
        batch_size: int = 128,
    ):
    
    if num_gpus is None:
        num_gpus = 1 if torch.cuda.is_available() else None
    
    system = ViTSystem(model_path, 
                       data_directory, 
                       batch_size)
    
    trainer = pl.Trainer(num_gpus=num_gpus, max_epochs=max_epochs)
    trainer.fit(system)

if __name__ == "__main__":
    typer.run(main)