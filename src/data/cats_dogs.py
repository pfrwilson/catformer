
import numpy as np
from torch.utils.data import Dataset, random_split
import pandas as pd
from torchvision import transforms
import os
import re
from tqdm import tqdm
from PIL import Image
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

class DogsVsCats(Dataset):

    summary_statistics = {
        'mean': np.array([0.48831898, 0.45508292, 0.41696587], dtype='float32'),
        'std': np.array([0.2601033, 0.25352228, 0.2561279], dtype='float32')
    }
    id2label = {
        0: 'cat', 
        1: 'dog'
    }

    def __init__(self, root, transform=None, target_transform=None, split='train'):
        
        assert split in ['train', 'val', 'test']
        self.split=split
        self.root=os.path.join(root, split)
        self.transform = transform
        self.target_transform = target_transform
        
        self.dataframe = pd.DataFrame(
            columns={
                'species': pd.Series(dtype=pd.StringDtype()), 
                'filename': pd.Series(dtype=pd.StringDtype()), 
                'sample_number': pd.Series(dtype=pd.Int32Dtype()),
                'label': pd.Series(pd.Int64Dtype())
            },
            index=pd.Index(range(len(os.listdir(self.root)))), 
        )
        
        with tqdm(
            os.listdir(self.root)
        ) as pbar:
            pbar.set_description(f'Loading annotation data. Split: {self.split}.')
            for i, filename in enumerate(pbar):

                species, number, ext = re.match('(\w+)\.(\d+)\.(\w+)', filename).groups()
                
                number = int(number)                          
                            
                label = 0 if species == 'cat' else (1 if species == 'dog' else np.NAN)
                
                self.dataframe.iloc[i] = {
                    'sample_number': number, 
                    'species': species,
                    'filename': filename,
                    'label': label,
                }

                if i % 100 == 0 :
                    pbar.set_postfix({'filename': filename})
                    
        self.dataframe = self.dataframe.sort_values(by=['species', 'sample_number'])
        self.dataframe = self.dataframe.reset_index(drop=True)

    def __getitem__(self, idx):
        
        try:
            label = self.dataframe['label'][idx]
            img_filepath = os.path.join(self.root, self.dataframe['filename'][idx])

            with open(img_filepath, 'rb') as f:
                img = Image.open(f)
                img.load()
        except KeyError:
            raise IndexError
        
        # apply transforms if applicable:
        if self.transform: 
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label
        

    def __len__(self):
        return len(self.dataframe)

    @staticmethod
    def get_default_augmentations(target_size):

        augmentations = transforms.RandomOrder([
            transforms.RandomResizedCrop(target_size),
            transforms.RandomApply(
                [transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)]
            ),
            transforms.RandomApply(
                [transforms.Grayscale(num_output_channels=3)]
            ),
            transforms.RandomHorizontalFlip()
        ])

        return augmentations

    @staticmethod
    def get_default_transform(target_size, use_augmentations=False):

        summary_statistics = DogsVsCats.summary_statistics
        mean, std = summary_statistics['mean'], summary_statistics['std']

        return transforms.Compose([
            DogsVsCats.get_default_augmentations((target_size, target_size))
            if use_augmentations else transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    
class DogsVsCatsDataModule(pl.LightningDataModule):
    
    def __init__(self, root, batch_size, image_size, num_workers=8):
        self.root=root 
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
    
    def setup(self, stage=None): 
         
        self.train_ds = DogsVsCats(
            self.root, 
            transform=DogsVsCats.get_default_transform(
                target_size=self.image_size, 
                use_augmentations=True
            ), 
            split='train'
        )
        
        self.val_ds = DogsVsCats(
            self.root, 
            transform=DogsVsCats.get_default_transform(
                target_size=self.image_size, 
                use_augmentations=False
            ), 
            split='val'
        )
        self.test_ds = DogsVsCats(
            self.root, 
            transform=DogsVsCats.get_default_transform(
                target_size=self.image_size, 
                use_augmentations=False
            ), 
            split='test'
        )
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, 
                          num_workers=self.num_dataloader_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, 
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.config.batch_size, 
                          num_workers=self.num_workers)
        
    
    