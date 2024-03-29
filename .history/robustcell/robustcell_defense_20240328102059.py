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

class scDefense():
    def __init__(self, adata, device='cpu', eps = 1, seed=2023):
        self.adata = adata # adata is the output of the model you plan to benchmark.
        self.device = device
        self.eps = eps
        self.seed = seed

    def fit_normal_nn(self, model, X_tr, X_test, y_tr, y_test):
        X_tr, X_val, y_tr, y_val = sklearn.model_selection.train_test_split(X_tr, y_tr, random_state=2023)
        X_tr, X_val, X_test, y_tr, y_val, y_test =torch.FloatTensor(X_tr),torch.FloatTensor(X_val),torch.FloatTensor(X_test),torch.FloatTensor(y_tr), torch.FloatTensor(y_val), torch.FloatTensor(y_test)
        train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
        valid_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        test_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)

        #ensure it is a model from pytorch_lightning
        print(model.log) #see log and log dir for model place

        lr_monitor = LearningRateMonitor(logging_interval='step')

        train_loader = DataLoader(train_dataset, batch_size=4096, num_workers=5, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=2048, num_workers=5)

        # train with both splits
        trainer = L.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=100)], max_epochs=1000)
        trainer.fit(model, train_loader, valid_loader)
        return model