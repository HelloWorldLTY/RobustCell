#This is one example for the neural network focusing on classification.
import torch
import torch.nn.functional as F
import lightning as L
import os
import pandas as pd
import numpy as np
import pickle
import sklearn.model_selection
import scipy.stats
import sklearn.metrics
import scanpy as sc

from torch.utils.data import DataLoader
from torch import nn
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


adata = sc.read_h5ad("/gpfs/gibbs/pi/zhao/tl688/scgpt_dataset/demo_train.h5ad")

layers =[3000, 512, len(set(adata.obs['Celltype']))]

# layers

class Encoder(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.shared_layers = nn.Sequential(nn.Linear(layers[0], layers[1]), 
                                nn.BatchNorm1d(layers[1]),
                                nn.ReLU(), 
                                nn.Dropout(input_dropout),
                                nn.Linear(layers[1], layers[1]),
                                nn.BatchNorm1d(layers[1]),
                                nn.ReLU(), 
                                nn.Dropout(input_dropout),
                                nn.Linear(layers[1], layers[2])
                               )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        return self.softmax (self.shared_layers(x))
        



class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        
        output_label = self.encoder(x)
        
        loss = self.loss(output_label, y.long())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        
        output_label = self.encoder(x)
        
        val_loss = self.loss(output_label, y.long())
        
        self.log("val_loss", val_loss.item())
        return val_loss

        
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        
        output_label = self.encoder(x)
        
        test_loss = self.loss(output_label, y.long())
        self.log("test_loss", test_loss.item())
        return test_loss
        
    def forward(self, x):
        return self.encoder(x)

    def configure_optimizers(self):
            optimizer = torch.optim.Adam(
                params=self.parameters(), 
                lr=eta
)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=10,
                verbose=True
            )
            return {
               'optimizer': optimizer,
               'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
               'monitor': 'val_loss'
           }


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

