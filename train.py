from logging.config import dictConfig
import hydra
from omegaconf import DictConfig
from src.systems.vit_system import ViTSystem
from src.data.cats_dogs import DogsVsCatsDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch

DEFAULT_DATA_ROOT = os.path.join(
    os.environ['HOME'], 
    'data', 
    'dogs-vs-cats',
    'data' 
)

DEFAULT_LOG_DIRECTORY = os.path.join(
    os.environ['HOME'],
    'lightning_logs'
)

DEFAULT_MODEL_PATH = 'google/vit-large-patch16-384'

@hydra.main(config_path='config', config_name='config')
def main(config: DictConfig):
    
    system = ViTSystem(config.model)
    
    datamodule = DogsVsCatsDataModule(
        root = config.data.root if config.data.root \
            else DEFAULT_DATA_ROOT, 
        image_size = config.model.image_size,
        batch_size=config.data.batch_size, 
        num_workers=config.data.num_workers, 
        use_augmentations_in_training=config.data.use_augmentations
    )

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else None,
        logger = TensorBoardLogger(
            config.training.logdir if config.training.logdir \
                else DEFAULT_LOG_DIRECTORY
        ), 
        max_epochs=config.training.max_epochs
    )
    
    trainer.fit(system, datamodule)
    trainer.validate(system, datamodule)

if __name__ == "__main__":
    main()