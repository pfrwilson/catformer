
import torch 
from torch.utils.data import Dataset

class TransformDataset(Dataset):
    """ 
    Wraps a dataset and applies a specified transform to its features
    """
    def __init__(self, dataset: Dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
    
    def __len__(self):
        return len(self.dataset)