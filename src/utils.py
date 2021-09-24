import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
import timm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ipyplot
from skimage import io
from sklearn.metrics import balanced_accuracy_score
from tqdm.auto import tqdm

class ImagesDataset(Dataset):
    def __init__(self, image_paths, labels = None, transform=None):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.targets = self.labels
        self.transform = transform
        
        if self.labels is not None:
            assert len(self.image_paths) == len(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = -1 # Carefully handle that!
        img = io.imread(self.image_paths[idx])
        img = img[...,:3] # Some images have 4 channels, fix that
        if self.transform:
            img = self.transform(img)

        return img, label

def evaluate_model(model, dataset, batch_size=32, num_workers=4):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=False)
    predictions = []
    labels = []
    with torch.no_grad():
        for x, y in tqdm(loader):
            prediction = model.predict(x).numpy()
            predictions += list(prediction)
            labels += list(y.numpy())
            
    return labels, predictions

