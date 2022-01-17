
from omegaconf import DictConfig
import hydra
from src.systems.vit_system import ViTSystem
import pytorch_lightning as pl
import os


DATA_DIRECTORY = os.path.join(
    os.getenv('~'), 
    'data', 
    'dogs-vs-cats', 
    'train'
)


@hydra.main(config_path = '.', config_name='config')
def main(config: DictConfig):
    
    system = ViTSystem(config.model.pretrained_transformer_name, 
                       DATA_DIRECTORY, 
                       config.train.batch_size)
    
    trainer = pl.Trainer()
    trainer.fit(system)

if __name__ == "__main__":
    main()