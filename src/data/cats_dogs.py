
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

    def __init__(self, root):
        self.root = root
        self.dataframe = pd.DataFrame()
        
        with tqdm(os.listdir(root)) as pbar:
            pbar.set_description('Loading annotation data')
            for i, filename in enumerate(pbar):

                species, number, ext = re.match('(\w+)\.(\d+)\.(\w+)', filename).groups()

                label = 0 if species == 'cat' else (1 if species == 'dog' else np.NAN)

                self.dataframe = self.dataframe.append({
                    'species': species,
                    'number': int(number),
                    'filename': filename,
                    'label': label,
                }, ignore_index=True)

                if i % 100 == 0 :
                    pbar.set_postfix({'filename': filename})

    def __getitem__(self, idx):
        try:
            label = self.dataframe['label'][idx]
            img_filepath = os.path.join(self.root, self.dataframe['filename'][idx])

            with open(img_filepath, 'rb') as f:
                img = Image.open(f)
                img.load()

            return img, label
        except KeyError:
            raise IndexError

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

#
#
#class DogVsCatsModule(pl.LightningDataModule):
#    
#    def __init__(self, image_size, root, batch_size=32):
#    
#        self.root = root
#        self.image_size = image_size
#        self.batch_size = batch_size
#        
#    def setup(self): 
#        
#        self.full_ds = DogsVsCats(self.root, shuffle=False)
#        
#        train_length = int(0.8 * len(self.full_ds))
#        test_length = len(self.full_ds) - train_length
#        
#        self.train_ds, self.val_ds = random_split(
#            self.full_ds,
#            [train_length, test_length], 
#        )
#        
#        # Apply augmentations by wrapping in map dataset
#        self.train_ds = MapDataset(
#            self.train_ds, 
#            DogsVsCats.get_default_transform(
#                target_size = self.image_size,
#                use_augmentations = True
#            )
#        )
#        self.val_ds = MapDataset(
#            self.val_ds, 
#            DogsVsCats.get_default_transform(
#                target_size = self.image_size, 
#                use_augmentations=False
#            )
#        )
#        
#    def train_dataloader(self):
#        return DataLoader(self.train_ds, batch_size=self.batch_size)
#    
#    def val_dataloader(self):
#        return DataLoader(self.val_ds, batch_size=self.batch_size)