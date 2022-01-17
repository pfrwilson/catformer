
from omegaconf import DictConfig
import hydra
from src.systems.vit_system import ViTSystem
import pytorch_lightning as pl

@hydra.main(config_path = '.', config_name='config')
def main(config: DictConfig):
    
    system = ViTSystem(config.model.pretrained_transformer_name, 
                       config.dataset.root_dir, 
                       config.train.batch_size)
    
    trainer = pl.Trainer()
    trainer.fit(system)

if __name__ == "__main__":
    main()