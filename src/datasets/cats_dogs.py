import einops
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
import os
import re
from tqdm import tqdm
from PIL import Image


class DogsVsCats(Dataset):

    def __init__(self, root, transform=None, target_transform=None, shuffle=True):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataframe = pd.DataFrame()
        self.summary_statistics = {
            'mean': np.array([0.48831898, 0.45508292, 0.41696587], dtype='float32'),
            'std': np.array([0.2601033, 0.25352228, 0.2561279], dtype='float32')
        }

        with tqdm(os.listdir(root)) as pbar:
            pbar.set_description('Loading annotation data')
            for filename in pbar:
                species, number, ext = re.match('(\w+)\.(\d+)\.(\w+)', filename).groups()
                label = 0. if species == 'cat' else (1. if species == 'dog' else np.NAN)
                self.dataframe = self.dataframe.append({
                    'species': species,
                    'number': int(number),
                    'filename': filename,
                    'label': label,
                }, ignore_index=True)
                pbar.set_postfix({'filename': filename})

        if shuffle:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, idx):
        try:
            label = self.dataframe['label'][idx]
            if self.target_transform:
                label = self.target_transform(label)

            img_filepath = os.path.join(self.root, self.dataframe['filename'][idx])

            with open(img_filepath, 'rb') as f:
                img = Image.open(f)
                img.load()

            if self.transform:
                img = self.transform(img)

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

    def get_default_transform(self, target_size, use_augmentations=False):

        summary_statistics = self.summary_statistics
        mean, std = summary_statistics['mean'], summary_statistics['std']

        return transforms.Compose([
            DogsVsCats.get_default_augmentations(target_size)
            if use_augmentations else transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    @staticmethod
    def get_default_target_transform():

        return transforms.Lambda(
            lambda label: F.one_hot(torch.tensor(int(label)), num_classes=2).float()
        )

