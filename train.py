import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch

from src.systems.vit_system import ViTSystem
from src.data.cats_dogs import DogsVsCatsDataModule

@hydra.main(config_path='conf', config_name='train')
def main(config: DictConfig):

    system = ViTSystem(config.model)
    if config.model.checkpoint:
        system = ViTSystem.load_from_checkpoint(config.model.checkpoint)
    
    datamodule = DogsVsCatsDataModule(config.data)

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else None,
        default_root_dir=config.training.logdir,
        max_epochs=config.training.max_epochs,
        check_val_every_n_epoch=1
    )
    
    if config.training.mode == 'train':
        trainer.fit(system, datamodule)
    
    if config.training.mode == 'val':
        trainer.validate(system, datamodule)
    
    if config.training.mode == 'test':
        trainer.test(system, datamodule)
    
if __name__ == "__main__":
    main()

