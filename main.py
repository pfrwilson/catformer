from logging.config import dictConfig
import hydra
from omegaconf import DictConfig
from src.systems.vit_system import ViTSystem
from src.data.cats_dogs import DogsVsCatsDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os
import torch

DEFAULT_MODEL_PATH = 'google/vit-large-patch16-384'


@hydra.main(config_path='config', config_name='config')
def main(config: DictConfig):

    system = ViTSystem(config.model)
    if config.training.ckpt_path:
        system = ViTSystem.load_from_checkpoint(config.training.ckpt_path)
    
    datamodule = DogsVsCatsDataModule(
        root=config.data.root,
        image_size=config.model.image_size,
        batch_size=config.data.batch_size, 
        num_workers=config.data.num_workers, 
        use_augmentations_in_training=config.data.use_augmentations
    )

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else None,
        default_root_dir=config.training.logdir,
        max_epochs=config.training.max_epochs,
        check_val_every_n_epoch=1
    )
    
    if config.training.mode == 'train':
        trainer.fit(system, datamodule,
                    ckpt_path=config.training.ckpt_path)
    
    if config.training.mode == 'val':
        trainer.validate(system, datamodule,
                    ckpt_path=config.training.ckpt_path)
        
if __name__ == "__main__":
    main()

