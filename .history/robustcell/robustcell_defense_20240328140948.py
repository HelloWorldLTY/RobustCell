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

import torch
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN,  GCNSVD, GCNJaccard, RGCN, ProGNN, SimPGCN
from deeprobust.graph.global_attack import Metattack

class scDefense():
    def __init__(self, adata, device='cpu', eps = 1, seed=2023):
        self.adata = adata # adata is the output of the model you plan to benchmark.
        self.device = device
        self.eps = eps
        self.seed = seed

    def fit_normal_nn(self, model, X_tr, X_test, y_tr, y_test, return_train = True):
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

        for m in model.encoder.modules():

            for child in m.children():

                if type(child) == nn.BatchNorm1d:

                    child.track_running_stats = False

                    child.running_mean = None

                    child.running_var = None
        model.encoder.eval()

        if return_train:
            with torch.no_grad():
                y_pred = model.encoder(X_tr)
                _ , y_pred = torch.max(y_pred, 1)
            
            print(sklearn.metrics.classification_report(y_tr.to(self.device), y_pred.to(self.device)))
        else:
            with torch.no_grad():
                y_pred = model.encoder(X_test)
                _ , y_pred = torch.max(y_pred, 1)
            
            print(sklearn.metrics.classification_report(y_test.to(self.device), y_pred.to(self.device)))

        return model

    def fit_normal_svm(self, model,  X_tr, X_test, y_tr, y_test, return_train = True):
        model.fit(X_tr, y_tr)

        if return_train:
            y_pred = model.predict(X_tr)
            
            print(sklearn.metrics.classification_report(y_tr, y_pred))
        else:
            y_pred = model.predict(X_test)
            
            print(sklearn.metrics.classification_report(y_test, y_pred))

        return model

    def adv_model_nn(self, model, X_clean, X_pos_train, X_pos_test, y_clean, y_pos_train, y_pos_test, return_train = False):
        X_tr = np.vstack([X_clean, X_pos_train])
        y_tr = np.vstack([y_clean, y_pos_train])
        X_test = X_pos_test
        y_test = y_pos_test

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

        for m in model.encoder.modules():

            for child in m.children():

                if type(child) == nn.BatchNorm1d:

                    child.track_running_stats = False

                    child.running_mean = None

                    child.running_var = None
        model.encoder.eval()

        if return_train:
            with torch.no_grad():
                y_pred = model.encoder(X_tr)
                _ , y_pred = torch.max(y_pred, 1)
            
            print(sklearn.metrics.classification_report(y_tr.to(self.device), y_pred.to(self.device)))
        else:
            with torch.no_grad():
                y_pred = model.encoder(X_test)
                _ , y_pred = torch.max(y_pred, 1)
            
            print(sklearn.metrics.classification_report(y_test.to(self.device), y_pred.to(self.device)))

        return model

    def adv_model_svm(self, model, X_clean, X_pos_train, X_pos_test, y_clean, y_pos_train, y_pos_test, return_train = False):

        X_tr = np.vstack([X_clean, X_pos_train])
        y_tr = np.vstack([y_clean, y_pos_train])
        X_test = X_pos_test
        y_test = y_pos_test
        model.fit(X_tr, y_tr)

        if return_train:
            y_pred = model.predict(X_tr)
            
            print(sklearn.metrics.classification_report(y_tr, y_pred))
        else:
            y_pred = model.predict(X_test)
            
            print(sklearn.metrics.classification_report(y_test, y_pred))

        return model
    
    def select_markergene(self, num_marker = 40, test_method = 'wilcoxon'):
        print("Please ensure that your data are normalzied.")
        adata = self.adata
        gene_list = adata.var_names

        sc.tl.rank_genes_groups(adata, groupby='celltype', n_genes=num_marker, method=test_method)
        emp_lit = []
        for i in adata.obs['celltype'].unique():
            emp_lit += list(adata.uns['rank_genes_groups']['names'][i])
        adata = adata[:, sorted(set(emp_lit))]

        return adata

class graphDefense():
    def __init__(self, adata, device='cpu', eps = 1, seed=2023):
        self.adata = adata # adata is the output of the model you plan to benchmark.
        self.device = device
        self.eps = eps
        self.seed = seed


    def run_GCNSVD(self, features, perturbed_adj, labels, idx_train, k=20):
        device = 'cpu'
        model = GCNSVD(nfeat=features.shape[1],
          nhid=16,
          nclass=labels.max().item() + 1,
          dropout=0.5, device=device ).to(device )
        model.fit(features, perturbed_adj.to(device), labels, idx_train, k)

        return model

    def run_GCNJaccard(self, features, perturbed_adj, labels, idx_train, threshold=0.03):
        device = 'cpu'
        model = GCNJaccard(nfeat=features.shape[1],
                nhid=16,
                nclass=labels.max().item() + 1,
                dropout=0.5, device='cpu').to('cpu')
        print(model)
        model.fit(features.numpy(), perturbed_adj.to(device).numpy(), labels, idx_train, threshold)

        return model

    def run_RGCN(self, features, perturbed_adj, labels, idx_train, nhid=32, train_iters=200):
        device = 'cpu'
        model = RGCN(nnodes=perturbed_adj.shape[0], nfeat=features.shape[1],
                    nclass=labels.max()+1, nhid, device=device)

        print(model)
        model.fit(features, perturbed_adj.to(device), labels, idx_train, train_iters)

        return model

    def run_SimPGCN(self, features, perturbed_adj, labels, idx_train, nhid=16, train_iters):
        model = SimPGCN(nnodes=features.shape[0], nfeat=features.shape[1],
        nhid=16, nclass=int(labels.max()+1), device=self.device)
        model = model.to(self.device)
        print(model)
        model.fit(features.to(self.device), perturbed_adj.to(self.device), labels, idx_train, train_iters)

        return model