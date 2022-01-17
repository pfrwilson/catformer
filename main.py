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
        max_epochs: Optional[int] = None, 
        model_path: str = DEFAULT_MODEL_PATH,
        batch_size: int = 128,
    ):
    
    system = ViTSystem(model_path, 
                       data_directory, 
                       batch_size)
    
    trainer = pl.Trainer(accelerator='auto', max_epochs=max_epochs)
    trainer.fit(system)

if __name__ == "__main__":
    typer.run(main)